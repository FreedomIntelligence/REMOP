import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # modeling
    untie_encoder: bool = field(
        default=False,
        metadata={"help": "no weight sharing between qry passage encoders"}
    )
    fine_tuning: bool = field(
        default=False,
        metadata={"help": "whether to fix plm parameters"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=1)

    # p*-tuning
    model_type: str = field(
        default="bert",
        metadata={
            "help": "The type of model, where we currently support bert, roberta"
        }
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    pre_seq_len: int = field(
        default=128,
        metadata={
            "help": "The length of prompt"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    general_prompt_path: str = field(
        default=None,
        metadata={
            "help": "path to general prompt file, to initialize training/inference mode"
        }
    )
    attribute_prompt_dir: str = field(
        default=None,
        metadata={
            "help": "path to attribute prompts' directory, the prompt files inside should be in format: lm_q_attribute_\{ATTRIBUTE_NAME\}.prompt or lm_p_attribute_\{ATTRIBUTE_NAME\}.prompt"
        }
    )
    prompt_weight: float = field(
        default=0.5,
        metadata={
            "help": "The prefix prompt weight used in the models"
        }
    )


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_n_passages: int = field(default=8)
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"})


    encode_in_path: str = field(default=None, metadata={"help": "Path to data to encode"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    encode_is_qry: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    q_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    p_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json') or f.endswith('jsonl')
            ]


@dataclass
class DenseTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})
    checkpoint_dir: str = field(default=None, metadata={"help": "path to checkpoint directory"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    general_lr: float = field(default=6e-3, metadata={"help": "learning rate of general prompt training"})
    attribute_lr: float = field(default=3e-3, metadata={"help": "learning rate of attribute prompt training"})

    general_weight_decay: float = field(default=0.0)
    attribute_weight_decay: float = field(default=0.0)