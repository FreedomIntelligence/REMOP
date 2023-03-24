import logging
import os
import sys
from contextlib import nullcontext

import datasets
from tqdm import tqdm

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)

from remop.arguments import ModelArguments, DataArguments, \
    DenseTrainingArguments as TrainingArguments
from remop.data import EncodeDataset, EncodeCollator
from remop.modeling import DenseOutput, DenseModelForInference


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    if training_args.local_rank in [-1, 0]:
        os.makedirs(data_args.encoded_save_path, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    num_labels = 1
    if os.path.isdir(model_args.model_name_or_path) and os.path.exists(os.path.join(model_args.model_name_or_path, "query_model")):
        config = AutoConfig.from_pretrained(
            os.path.join(model_args.model_name_or_path, "query_model"),
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
        )
    # p*-tuning
    config.fine_tuning = model_args.fine_tuning
    config.prefix = model_args.prefix
    config.pre_seq_len = model_args.pre_seq_len
    config.hidden_dropout_prob = model_args.hidden_dropout_prob

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = DenseModelForInference.build(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    text_max_length = data_args.q_max_len if data_args.encode_is_qry else data_args.p_max_len
    encode_dataset = EncodeDataset(data_args.encode_in_path, tokenizer, max_len=text_max_length, prompt_weight=model_args.prompt_weight)
    dataset_attributes = encode_dataset.dataset_attributes

    if training_args.world_size > 1:
        encode_dataset.encode_data = encode_dataset.encode_data.shard(training_args.world_size, training_args.local_rank)

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=EncodeCollator(
            tokenizer,
            max_length=text_max_length,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    encoded = []
    lookup_indices = []
    model.load_prompt(model_args.general_prompt_path, model_args.attribute_prompt_dir, model_args.untie_encoder, dataset_attributes)
    model = model.to(training_args.device)
    model.eval()

    pbar = tqdm(encode_loader, desc="encode data") if training_args.local_rank in [-1, 0] else encode_loader
    for (batch_ids, batch, attrs) in pbar:
        lookup_indices.extend(batch_ids)
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)
                if data_args.encode_is_qry:
                    model_output: DenseOutput = model(query=batch, attrs=attrs)
                    encoded.append(model_output.q_reps.cpu())
                else:
                    model_output: DenseOutput = model(passage=batch)
                    encoded.append(model_output.p_reps.cpu())

    encoded = torch.cat(encoded)
    torch.save((encoded, lookup_indices), os.path.join(data_args.encoded_save_path, f"part-{training_args.local_rank}.pt"))
    if dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    main()
