import logging
import os
import sys

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from remop.arguments import (
    ModelArguments,
    DataArguments,
    DenseTrainingArguments as TrainingArguments,
)
from remop.data import TrainDataset, QPCollator
from remop.modeling import DenseModel
from remop.trainer import DenseTrainer as Trainer, GCTrainer
import wandb

wandb.init(project="train_rempo")

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        num_labels=model_args.projection_out_dim,
    )
    # p*-tuning
    config.fine_tuning = model_args.fine_tuning
    config.prefix = model_args.prefix
    config.pre_seq_len = model_args.pre_seq_len
    config.hidden_dropout_prob = model_args.hidden_dropout_prob
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    # we need to set the model_name_or_path here because the build function will use it to load the model
    model_args.model_name_or_path = training_args.checkpoint_dir
    model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    assert (
        data_args.train_dir is not None
    ), "In training process: data_args.train_dir should not be None"
    train_dataset = TrainDataset(data_args, data_args.train_path, tokenizer)
    dataset_attributes = train_dataset.dataset_attributes
    trainer_cls = GCTrainer if training_args.grad_cache else Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QPCollator(
            tokenizer, max_p_len=data_args.p_max_len, max_q_len=data_args.q_max_len
        ),
    )

    attribute_prompt = (
        model_args.attribute_prompt_dir
        if model_args.attribute_prompt_dir
        else len(dataset_attributes)
    )
    trainer.load_prompt(
        model_args.general_prompt_path,
        attribute_prompt,
        model_args.untie_encoder,
        dataset_attributes,
    )
    train_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=training_args.checkpoint_dir)
    trainer.save_prompt(dataset_attributes)
    trainer.save_model()

    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
