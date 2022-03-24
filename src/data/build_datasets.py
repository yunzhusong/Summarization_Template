""" Builds datasets. """
import logging
import os
from datasets import load_dataset, concatenate_datasets

import pdb

summarization_name_mapping = {
    "xsum": ("document", "summary"),
    "cnn_dailymail": ("article", "highlights"),
    "newsroom": ("text", "summary"),  # require manual files 
    "multi_news": ("document", "summary"),
    "gigaword": ("document", "summary"),
    "wikihow": ("text", "headline"),  # require manual files
    "reddit": ("content", "summary"),
    "reddit_tifu": ("documents", "tldr"),
    "big_patent": ("description", "abstract"),
    "scientific_papers": ("article", "abstract"),
    "aeslc": ("email_body", "subject_line"),
    "billsum": ("text", "summary"),
    "xsum_own": ("document", "summary"),
    "cnn_dailymail_own": ("article", "highlights"),
    "reddit_own": ("content", "summary"),
}

summarization_length_mapping = {
    "xsum": (1024, 64),
    "cnn_dailymail": (1024, 128),
    "newsroom": (1024, 128),
    "multi_news": (1024, 256),
    "gigaword": (1024, 32),
    "wikihow": (1024, 256),
    "reddit": (1024, 128),
    "reddit_tifu": (1024, 128),
    "big_patent": (1024, 256),
    "scientific_papers": (1024, 256),
    "aeslc": (1024, 32),
    "billsum": (1024, 256),
    "xsum_own": (1024, 64),
    "cnn_dailymail_own": (1024, 128),
    "reddit_own": (1024, 128),
}

summarization_own_file_mapping = {
    "xsum_own":
        "../../SummaryAugment/datasets/fusion/xsum_fusion_c100_two_sents_each",
    "cnn_dailymail_own":
        "../../SummaryAugment/datasets/fusion/cnndm_fusion_x100_two_sents_each",
    "reddit_own":
        "../../SummaryAugment/datasets/pattern/reddit_pattern_x10_40K",
}

logger = logging.getLogger(__name__)

def build_datasets(data_args,
                   training_args,
                   tokenizer,
                   ignore_train=False,
                   ignore_eval=False):

    if data_args.dataset_name is not None and data_args.dataset_name[
            -3:] != 'own':
        # Download and load a dataset from the hub.
        if data_args.dataset_name == 'cnn_dailymail':
            data_args.dataset_config_name = '3.0.0'
        elif data_args.dataset_name == 'reddit_tifu':
            data_args.dataset_config_name = 'long'
        elif data_args.dataset_name == 'wikihow':
            data_args.dataset_config_name = 'all'
        elif data_args.dataset_name == 'arxiv':
            data_args.dataset_name = 'scientific_papers'
            data_args.dataset_config_name = 'arxiv'
        elif data_args.dataset_name == 'pubmed':
            data_args.dataset_name = 'scientific_papers'
            data_args.dataset_config_name = 'pubmed'
        else:
            data_args.dataset_config_name = None

        if data_args.dataset_name == 'wikihow':
            datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                data_dir=os.path.abspath("../cache/manual/wikihow"))
        elif data_args.dataset_name == 'newsroom':
            datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                data_dir=os.path.abspath("../cache/manual/newsroom"))
        else:
            datasets = load_dataset(data_args.dataset_name,
                                    data_args.dataset_config_name)
    else:
        # Use your own dataset
        data_dir = summarization_own_file_mapping.get(data_args.dataset_name,
                                                      None)
        if os.path.isfile(os.path.join(data_dir, 'train.csv')):
            data_args.train_file = os.path.join(data_dir, 'train.csv')
        if os.path.isfile(os.path.join(data_dir, 'validation.csv')):
            data_args.validation_file = os.path.join(data_dir, 'validation.csv')
        if os.path.isfile(os.path.join(data_dir, 'test.csv')):
            data_args.test_file = os.path.join(data_dir, 'test.csv')

        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)

    # NEW: shuffle datasets before select
    if data_args.shuffle_before_select:
        datasets = datasets.shuffle(seed=0)

    # [For NEWSROOM]
    # Only use data that is abstractive (as PEGASUS)
    if data_args.dataset_name == 'newsroom':
        logger.warning("Extract abstractive data for NEWSROOM dataset...")
        datasets = datasets.filter(
            lambda example: example['density_bin'] == 'abstractive')

    # [For Reddit/Reddit-TIFU]
    # Only have training examples, manually split in 80/10/10
    if data_args.dataset_name in ('reddit', 'reddit_tifu'):
        logger.warning("Make train/valid/test splits for Reddit(-TIFU) dataset...")

        train_data_num = datasets.num_rows['train']
        first_sep_point = int(train_data_num * 0.8)
        second_sep_point = first_sep_point + int(train_data_num * 0.1)

        train_dataset = datasets['train'].select(range(0, first_sep_point))
        val_dataset = datasets['train'].select(range(first_sep_point, second_sep_point))
        test_dataset = datasets['train'].select(range(second_sep_point, train_data_num))
        datasets['train'] = train_dataset
        datasets['validation'] = val_dataset
        datasets['test'] = test_dataset

    # Take column names from datasets
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        raise "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name,
                                                     None)
    if data_args.text_column is None:
        text_column = dataset_columns[
            0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[
            1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # NOTE: Get max source length and max target length for the corpus
    max_source_length, max_target_length = summarization_length_mapping.get(
        data_args.dataset_name, None)
    data_args.max_source_length = max_source_length
    data_args.max_target_length = max_target_length
    data_args.val_max_target_length = max_target_length
    data_args.train_val_max_target_length = max_target_length

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    padding = "max_length" if data_args.pad_to_max_length else False

    # Function for tokenization
    def preprocess_function(examples):

        inputs = examples[text_column]
        targets = examples[summary_column]

        # NOTE: fix none input
        inputs = [inp if inp else " " for inp in inputs]

        # Add prefix
        inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs,
                                 max_length=max_source_length,
                                 padding=padding,
                                 truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets,
                               max_length=max_target_length,
                               padding=padding,
                               truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100
        # when we want to ignore padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100)
                                    for l in label]
                                   for label in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if training_args.do_train and not ignore_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            # NEW: start from specified data indice
            if data_args.select_start_indice:
                start_indice = data_args.select_start_indice
                end_indice = start_indice + data_args.max_train_samples
                train_dataset = train_dataset.select(
                    range(start_indice, end_indice))
            else:
                train_dataset = train_dataset.select(
                    range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval and not ignore_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict and not ignore_eval:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(
                data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    return train_dataset, eval_dataset, test_dataset
