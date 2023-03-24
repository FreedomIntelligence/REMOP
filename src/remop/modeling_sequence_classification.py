import torch
import torch.nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, ModuleList

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .modeling_prefix_encoder import PrefixEncoder

import logging

logger = logging.getLogger(__name__)

class BertPrefixForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        if not config.fine_tuning:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.general_prefix_encoder = PrefixEncoder(config)
        
        self.attribute_prefix_encoders = ModuleList()
        
        plm_param = 0
        for name, param in self.bert.named_parameters():
            plm_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        non_plm_param = all_param - plm_param
        logger.info(f"All param size: {all_param}; Plm param size: {plm_param}; Non-Plm param size: {non_plm_param}")

    def get_prompt(self, batch_size, attrs):
        pkv_size = (
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )

        prefix_tokens = self.prefix_tokens.unsqueeze(0).to(self.bert.device)
        general_past_key_values = self.general_prefix_encoder(prefix_tokens).view(pkv_size)

        if attrs is None or len(self.attribute_prefix_encoders) == 0:
            past_key_values = general_past_key_values.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        else:
            assert batch_size == attrs.shape[0], "Wrong attributes shape: {}, batch_size: {}".format(attrs.shape, batch_size)
            assert len(self.attribute_prefix_encoders) == attrs.shape[1], "Incompatiable attribute size, number of attribute prompt: {}, number of attributes: {}".format(len(self.attribute_prefix_encoders), attrs.shape[1])

            # attrs shape: (batch_size, n_prompt)
            attrs = attrs.to(self.bert.device)
            attribute_past_key_values = [attribute_encoder(prefix_tokens) for attribute_encoder in self.attribute_prefix_encoders]
            
            n_prompt = len(attribute_past_key_values)
            attribute_past_key_values = [pkv.view(pkv_size) for pkv in attribute_past_key_values]

            # pkvs shape: (n_prompt, pre_seq_len, n_layer*2, n_head, n_embd)
            pkvs = torch.cat(attribute_past_key_values, dim=0) 

            # past_key_values: (batch_size, pre_seq_len, n_layer*2, n_head, n_embd)
            attribute_past_key_values = torch.matmul(attrs, pkvs.view(n_prompt, -1)).view((batch_size, *pkv_size))

            past_key_values = general_past_key_values + attribute_past_key_values
        
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    
    def load_attribute_prompts(self, state_dicts):
        logging.info("BertPrefixForSequenceClassification loading attribute prompts, current: {}, load_dicts: {}".format(len(self.attribute_prefix_encoders), len(state_dicts)))
        for state_dict in state_dicts:
            prefix_encoder = PrefixEncoder(self.config)
            prefix_encoder.load_state_dict(state_dict)
            self.attribute_prefix_encoders.append(prefix_encoder)

    def get_attribute_prompts(self):
        return [prefix_encoder.state_dict() for prefix_encoder in self.attribute_prefix_encoders]

    def init_attribute_prompts(self, n_prompts):
        self.attribute_prefix_encoders = ModuleList()
        for _ in range(n_prompts):
            self.attribute_prefix_encoders.append(PrefixEncoder(self.config).to(self.bert.device))
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        attrs=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size, attrs=attrs)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaPrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        if not config.fine_tuning:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.general_prefix_encoder = PrefixEncoder(config)
        self.attribute_prefix_encoders = ModuleList()

    def get_prompt(self, batch_size, attrs):
        pkv_size = (
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        prefix_tokens = self.prefix_tokens.unsqueeze(0).to(self.roberta.device)
        general_past_key_values = self.general_prefix_encoder(prefix_tokens).view(pkv_size)

        if attrs is None or len(self.attribute_prefix_encoders) == 0:
            past_key_values = general_past_key_values.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        else:
            assert batch_size == attrs.shape[0], "Wrong attributes shape: {}, batch_size: {}".format(attrs.shape, batch_size)
            assert len(self.attribute_prefix_encoders) == attrs.shape[1], "Incompatiable attribute size, number of attribute prompt: {}, number of attributes: {}".format(len(self.attribute_prefix_encoders), attrs.shape[1])

            # attrs shape: (batch_size, n_prompt)
            attrs = attrs.to(self.roberta.device)
            attribute_past_key_values = [attribute_encoder(prefix_tokens) for attribute_encoder in self.attribute_prefix_encoders]

            n_prompt = len(attribute_past_key_values)
            attribute_past_key_values = [pkv.view(pkv_size) for pkv in attribute_past_key_values]

            # pkvs: (n_prompt, pre_seq_len, n_layer*2, n_head, n_embd)
            pkvs = torch.cat(attribute_past_key_values, dim=0)
            
            # past_key_values: (batch_size, pre_seq_len, n_layer*2, n_head, n_embd)
            attribute_past_key_values = torch.matmul(attrs, pkvs.view(n_prompt, -1)).view((batch_size, *pkv_size))
            
            past_key_values = general_past_key_values + attribute_past_key_values

        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def load_attribute_prompts(self, state_dicts):
        for state_dict in state_dicts:
            prefix_encoder = PrefixEncoder(self.config)
            prefix_encoder.load_state_dict(state_dict)
            self.attribute_prefix_encoders.append(prefix_encoder)

    def get_attribute_prompts(self):
        return [prefix_encoder.state_dict() for prefix_encoder in self.attribute_prefix_encoders]

    def init_attribute_prompts(self, n_prompts):
        self.attribute_prefix_encoders = ModuleList()
        for _ in range(n_prompts):
            self.attribute_prefix_encoders.append(PrefixEncoder(self.config).to(self.roberta.device))
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        attrs=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size, attrs=attrs)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
