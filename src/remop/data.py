import os
import random
from dataclasses import dataclass
from typing import Union, List

import torch
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import DenseTrainer

import logging
logger = logging.getLogger(__name__)

# TrainDataset for training, can jointly train multiple datasets by using List[str] in path_to_data
# The dataset from BEIR cannot be used directly because attribute labels for each data are required. 
class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            path_to_data: Union[str, List[str], datasets.Dataset],
            tokenizer: PreTrainedTokenizer,
            trainer: DenseTrainer = None,
    ):
        logging.info("Path_to_data: {}".format(path_to_data))
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            cache_dir="./cache"
        )['train']

        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)
        self.dataset_attributes = self.attribute_preprocessing()

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def attribute_preprocessing(self):
        attribute_dict = {}
        for item in self.train_data:
            attrs = item.get('attributes', [])
            for attr in attrs:
                attribute_dict[attr] = attribute_dict.get(attr, 0) + 1
        
        logging.info("Data Length: {}".format(self.total_len))
        logging.info("Data Attribute length: {}, dict: {}".format(len(attribute_dict), attribute_dict))

        prompt_ids = []
        dataset_attributes = list(attribute_dict.keys())
        for item in self.train_data:
            attrs = item.get('attributes', [])
            ids = [1.0 if attr in attrs else 0 for attr in dataset_attributes]
            
            # simply combine attribute prompts by average aggregation
            attrs_len = float(len([i for i in ids if i != 0]))
            if attrs_len != 0:
                ids = [i / attrs_len for i in ids]

            prompt_ids.append(ids)
        self.train_data = self.train_data.add_column("prompt_ids", prompt_ids)
        return dataset_attributes

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> [BatchEncoding, List[BatchEncoding], torch.Tensor]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['pos_ctxs']
        group_negatives = group['neg_ctxs']
        attrs = group['prompt_ids']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        encoded_passages.append(self.create_one_example(pos_psg))
        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))

        return encoded_query, encoded_passages, attrs


class EncodeDataset(Dataset):
    input_keys = ['_id', 'text', 'prompt_ids']

    def __init__(self, path_to_json: Union[List[str], datasets.Dataset], tokenizer: PreTrainedTokenizer, max_len=128, prompt_weight=0.5):
        if isinstance(path_to_json, datasets.Dataset):
            self.encode_data = path_to_json
        else:
            if os.path.isdir(path_to_json):
                data_files = [os.path.join(path_to_json, de) for de in os.listdir(path_to_json)]
            else:
                data_files = [path_to_json]
            self.encode_data = datasets.load_dataset(
                'json',
                data_files=data_files,
                cache_dir="./cache"
            )['train']

        self.tok = tokenizer
        self.max_len = max_len
        self.total_len = len(self.encode_data)
        self.prompt_weight = prompt_weight
        self.dataset_attributes = self.attribute_preprocessing()
        
    def attribute_preprocessing(self):
        attribute_dict = {}
        for item in self.encode_data:
            attrs = item.get('attributes', [])
            for attr in attrs:
                attribute_dict[attr] = attribute_dict.get(attr, 0) + 1
        
        logging.info("Data Length: {}".format(self.total_len))
        logging.info("Data Attribute length: {}, dict: {}".format(len(attribute_dict), attribute_dict))

        prompt_ids = []
        dataset_attributes = list(attribute_dict.keys())
        for item in self.encode_data:
            attrs = item.get('attributes', [])
            ids = [self.prompt_weight if attr in attrs else 0 for attr in dataset_attributes]

            # simple average aggregation, when no module weights is given
            attrs_len = float(len([i for i in ids if i != 0]))
            if attrs_len != 0:
                ids = [i / attrs_len for i in ids]

            prompt_ids.append(ids)
        
        self.encode_data = self.encode_data.add_column("prompt_ids", prompt_ids)
        return dataset_attributes

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> [str, BatchEncoding, List]:
        text_id, text, attrs = (self.encode_data[item][f] for f in self.input_keys)

        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text, attrs


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg, attrs]] to List[qry], List[psg], Tensor[attrs]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        aa = [f[2] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        a_collated = torch.tensor(aa).float()

        return q_collated, d_collated, a_collated 


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        text_attributes = [x[2] for x in features]
        collated_features = super().__call__(text_features)

        return text_ids, collated_features, torch.tensor(text_attributes).float()
