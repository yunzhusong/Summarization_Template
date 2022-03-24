""" Arguments for running. """
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments


@dataclass
class CustomSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Customized sequence-to-sequence training arguments.
    """

    # Overwrite TrainingArguments default value
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help":
                "Remove columns not required by the model when using an nlp.Dataset."
        },
    )

    # Set generation max length for evaluation during training
    train_val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
                "This argument is used to override the ``max_length`` param of ``model.generate``, which is used "
                "in the evaluation() function calls during _maybe_log_save_evaluate()."
        },
    )

    insert_adapters: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to insert adapters, this argument will make only adapters trainable. "
        },
    )
    save_model_accord_to_rouge: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to save model according to ROUGE-1 score instead of loss."
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help":
                "Path to pretrained model or model identifier from huggingface.co/models"
        })
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list."
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Pretrained config name or path if not the same as model_name"
        })
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Pretrained tokenizer name or path if not the same as model_name"
        })
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help":
                "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
        },
    )

    # NEW: load_trained_model
    load_trained_model_from: Optional[str] = field(
        default=False,
        metadata={
            "help":
                "Specified a pre-trained model path, this argument only load the model checkpoint."
        })

    load_model_before_insert_adapter: bool = field(
        default=False, metadata={"help": "Load model before insert adapter"})

    # NEW: adapter
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "The hidden size for adapter."},
    )
    adapter_act: Optional[str] = field(
        default="relu",
        metadata={"help": "The activation function for adapter"},
    )
    adapter_init_range: Optional[float] = field(
        default=1e-4,
        metadata={"help": "The initialization range for adapter."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library)."
        })
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The configuration name of the dataset to use (via the datasets library)."
        })
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "The input training data file (a jsonlines or csv file)."
        })
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "An optional input evaluation data file to evaluate the metrics (rouge) on "
                "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "An optional input test data file to evaluate the metrics (rouge) on "
                "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of processes to use for the preprocessing."
        },
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help":
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help":
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help":
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help":
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "A prefix to add before every source text (useful for T5 models)."
        })

    # NEW
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "The name of the validation dataset to use (via the datasets library)."
        })

    shuffle_before_select: bool = field(
        default=True,
        metadata={
            "help":
                "Whether to shuffle the dataset before select data. This argument works for all dataset splits."
        })

    select_start_indice: Optional[int] = field(
        default=0,
        metadata={
            "help":
                "The first data indice for selection. This argument only works for training set."
        })

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"
                ], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
