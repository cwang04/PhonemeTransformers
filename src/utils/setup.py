""" Utilities for setting up experiments from config """

import logging
import os
import random
from typing import Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

import wandb
from src import config

# A logger for this file
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility

    Args:
        seed (int): seed to set
    """

    if seed < 0:
        logger.warning("Skipping seed setting for reproducibility")
        logger.warning("If you would like to set a seed, set seed to a positive value in config")
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    cfg: config.DatasetParams,
) -> Union[datasets.Dataset, datasets.DatasetDict, datasets.IterableDataset, datasets.IterableDatasetDict]:
    """Loads dataset from config

    Args:
        cfg (config.DatasetParams): hydra config object
    Returns:
        datasets.Dataset: loaded dataset
    """

    dataset = datasets.load_dataset(
        cfg.name,
        cfg.subconfig,
        token=os.environ["HF_READ_TOKEN"],
    )

    # Drop rows where target_child_age is none or is larger than the max_age
    if cfg.max_age is not None:
        if "target_child_age" not in dataset["train"].column_names:
            raise ValueError(f"max_age set to {cfg.max_age} but dataset does not contain target_child_age column")
        dataset = dataset.filter(
            lambda x: x["target_child_age"] is not None and x["target_child_age"] <= cfg.max_age,
            num_proc=(64 if torch.cuda.is_available() else 1),
        )
        logger.info(f"Filtered dataset to only include examples with target_child_age <= {cfg.max_age}")

    # Drop rows where is_child is True
    if cfg.remove_child_utterances is not None and cfg.remove_child_utterances:
        if "is_child" not in dataset["train"].column_names:
            raise ValueError(f"remove_child_utterances set to {cfg.remove_child_utterances} but dataset does not contain is_child column")
        dataset = dataset.filter(lambda x: x["is_child"] is False, num_proc=(64 if torch.cuda.is_available() else 1))
        logger.info("Filtered dataset to only include examples where is_child is False")

    # If dataset has no 'valid' split, create it
    if "valid" not in dataset.keys():
        num_valid_examples = cfg.valid_size
        logger.info(f"Creating validation split of {num_valid_examples} examples...")
        # Create the validation split by sampling evenly from the train dataset
        num_train_examples = len(dataset["train"])
        valid_indices = np.linspace(0, num_train_examples, num_valid_examples, endpoint=False, dtype=int)
        valid_indices = valid_indices % num_train_examples
        valid_indices = np.sort(valid_indices)
        dataset["valid"] = dataset["train"].select(valid_indices)
        indices_to_keep = [i for i in range(len(dataset["train"])) if i not in valid_indices]
        dataset["train"] = dataset["train"].select(indices_to_keep)
        logger.info("Succesfully created validation split")

    dataset = dataset.remove_columns(set(dataset["train"].column_names) - set([cfg.text_column]))

    # Rename target column to "text"
    if cfg.text_column != "text":
        dataset = dataset.rename_column(cfg.text_column, "text")

    return dataset


def setup_wandb(cfg: config.ExperimentParams) -> None:
    """Setup wandb for logging

    Args:
        cfg (config.ExperimentParams): hydra config object
    """

    wandb_entity = os.environ["WANDB_ENTITY"]
    os.environ["WANDB_PROJECT"] = cfg.group
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.resume_checkpoint_path:
        os.environ["WANDB_RUN_ID"] = cfg.resume_run_id
        os.environ["WANDB_RESUME"] = "allow"
    wandb.init(
        entity=wandb_entity,
        project=cfg.group,
        name=cfg.name,
        config=wandb.config,
        resume="allow",
        id=cfg.resume_run_id,
    )


def subsample_dataset_pre_tokenized(dataset: datasets.Dataset, num_examples: int, subsample_type: Optional[str]) -> datasets.Dataset:
    """Subsamples the dataset based on the subsample_type, before preprocessing has occurred

    Args:
        dataset (datasets.Dataset): dataset to subsample
        num_examples (int): number of examples to subsample
        subsample_type (Optional[str]): type of subsampling to perform. Must be one of "examples", "words", or "tokens".

    Returns:
        datasets.Dataset: subsampled dataset
    """

    if subsample_type is None or subsample_type == "examples":
        return dataset.shuffle().select(range(num_examples))
    elif subsample_type == "words":
        sampled_dataset = dataset.shuffle()
        cumulative_word_count = 0
        for i, example in enumerate(list(sampled_dataset["text"])):
            cumulative_word_count += example.count("WORD_BOUNDARY")  # TODO: This might not always be the word boundary token
            if cumulative_word_count >= num_examples:
                return sampled_dataset.select(range(i))
        raise RuntimeError(f"Subsample count {num_examples} exceeds total word count in dataset: {cumulative_word_count}")
    elif subsample_type == "tokens":
        raise RuntimeError(f"Cannot use subsample_type='tokens' when subsampling before tokenization")
    else:
        raise RuntimeError(f'Invalid value "{subsample_type}" for subsample_type. Must be one of "examples", "tokens" or "words".')


def subsample_dataset(dataset: datasets.Dataset, num_examples: int, subsample_type: Optional[str]) -> datasets.Dataset:
    """Subsamples the dataset based on the subsample_type

    Args:
        dataset (datasets.Dataset): dataset to subsample
        num_examples (int): number of examples to subsample
        subsample_type (Optional[str]): type of subsampling to perform. Must be one of "examples", "words", or "tokens".

    Returns:
        datasets.Dataset: subsampled dataset
    """

    if subsample_type is None or subsample_type == "examples":
        return dataset.shuffle().select(range(num_examples))
    elif subsample_type == "words":
        sampled_dataset = dataset.shuffle()
        cumulative_word_count = 0
        for i, example in enumerate(list(sampled_dataset["word_starts"])):
            cumulative_word_count += example.count(1)
            if cumulative_word_count >= num_examples:
                return sampled_dataset.select(range(i))
        raise RuntimeError(f"Subsample count {num_examples} exceeds total word count in dataset: {cumulative_word_count}")
    elif subsample_type == "tokens":
        sampled_dataset = dataset.shuffle()
        cumulative_token_count = 0
        for i, example in enumerate(list(sampled_dataset["input_ids"])):
            cumulative_token_count += len(example)
            if cumulative_token_count >= num_examples:
                return sampled_dataset.select(range(i))
        raise RuntimeError(f"Subsample count {num_examples} exceeds total token count in dataset: {cumulative_token_count}")
    else:
        raise RuntimeError(f'Invalid value "{subsample_type}" for subsample_type. Must be one of "examples", "tokens" or "words".')


def load_tokenizer(cfg: config.TokenizerParams) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Sets up tokenizer for the model, based on tokenizer configurations

    Args:
        cfg (TransformerSegmentationConfig): hydra config object
    """

    full_tokenizer_name = cfg.name

    # anything that's not name and vocab_size is an optional tokenizer kwarg
    remove_keys = ["name", "vocab_size"]
    tokenizer_kwargs = {key: val for key, val in cfg.items() if key not in remove_keys and val is not None}

    tokenizer = AutoTokenizer.from_pretrained(
        full_tokenizer_name,
        **tokenizer_kwargs,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    return tokenizer
