import os
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
# from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss

import logging
logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache
    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False


class DenseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(DenseTrainer, self).__init__(*args, **kwargs)
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        logger.info("get_train_dataloader train_batch_size: {}, per_device_train_batch_size: {}".format(self.args.train_batch_size, self.args.per_device_train_batch_size))
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, num_items_in_batch=None):
        query, passage, attrs = inputs
        loss = model(query=query, passage=passage, attrs=attrs).loss
        return loss

    def training_step(self, *args):
        return super(DenseTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    # Save general prompt and attribute prompts respectively
    def save_prompt(self, dataset_attributes, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir else self.args.output_dir
        output_dir = os.path.join(output_dir, "trained_prompts")
        os.makedirs(output_dir, exist_ok=True)

        state_dict = self.model.state_dict()
        general_state_dict = dict()
        for key, value in state_dict.items():
            if 'general_prefix_encoder' in key:
                general_state_dict[key] = value
        torch.save(general_state_dict, os.path.join(output_dir, "general_state_dict.prompt"))
        
        lm_q_prompts, lm_p_prompts = self.model.get_attribute_prompts()
        for index, state_dict in enumerate(lm_q_prompts):
            torch.save(state_dict, os.path.join(output_dir, "lm_q_attribute_{}.prompt".format(dataset_attributes[index])))
        for index, state_dict in enumerate(lm_p_prompts):
            torch.save(state_dict, os.path.join(output_dir, "lm_p_attribute_{}.prompt".format(dataset_attributes[index])))

        logger.info("Saved prompt in %s", output_dir)

    # Load general prompt from file and load attribute prompts from directory
    def load_prompt(self, general_prompt_path, attribute_prompt, untie_encoder: bool, dataset_attributes):
        self.model.load_prompt(general_prompt_path, attribute_prompt, untie_encoder, dataset_attributes)

    # Override create_optimizer to set general prompt and attribute promots learning rate
    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters() if n in decay_parameters and "prefix_encoder" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if n not in decay_parameters and "prefix_encoder" not in n],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if "general_prefix_encoder" in n],
                    "lr": self.args.general_lr,
                    "weight_decay": self.args.general_weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters() if "attribute_prefix_encoder" in n],
                    "lr": self.args.attribute_lr,
                    "weight_decay": self.args.attribute_weight_decay,
                }
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            #     self.optimizer = OSS(
            #         params=optimizer_grouped_parameters,
            #         optim=optimizer_cls,
            #         **optimizer_kwargs,
            #     )
            # else:
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        # if is_sagemaker_mp_enabled():
        #     self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 2
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    attr_key = list(model_input.keys())[1]
    attr_val = model_input[attr_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_attrs = attr_val.split(chunk_size, dim=0)
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c, attr_key: attr_v} for c, attr_v in zip(chunked_arg_val, chunked_attrs)]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(DenseTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError('Grad Cache package not available.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls(self.model.data_args.train_n_passages)
        self.scaler = torch.cuda.amp.GradScaler() if self.args.fp16 else None


        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler
        )

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()

        queries, passages, attrs = self._prepare_inputs(inputs)
        n_passages = self.model.data_args.train_n_passages
        queries, passages = {'query': queries, 'attrs': attrs}, {'passage': passages, 'attrs': torch.cat([i.repeat(n_passages, 1) for i in attrs], dim=0)}
        
        # _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages)

        return loss
