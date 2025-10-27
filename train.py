"""Train language models on phonemized datasets."""

import logging
import os

# config-related imports
import hydra
import torch

# training pipeline imports
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers.data import DataCollatorForLanguageModeling
from transformers.trainer import TrainerState, TrainingArguments

# wandb for logging metrics
import wandb

# local imports
from src.config import TransformerSegmentationConfig
from src.datacollator import CustomDataCollatorForLanguageModeling
from src.models import load_model
from src.preprocessing import DataPreprocessor
from src.trainer import CustomTrainer
from src.utils import setup

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=TransformerSegmentationConfig)

# A logger for this file
logger = logging.getLogger(__name__)

DRY_RUN_SUBSAMPLE_FACTOR = 10 // (10 if torch.cuda.device_count() > 1 else 1)
DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10


def check_and_set_environment_variables(cfg: TransformerSegmentationConfig) -> None:
    """Checks huggingface tokens exist and sets up wandb environment variables"""

    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables. Check .env file and source it if necessary."

    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        if cfg.experiment.resume_run_id is not None:
            raise RuntimeError("resume_run_id is set but offline_run is True. Ignoring resume_run_id.")
    else:
        assert (
            "WANDB_ENTITY" in os.environ
        ), "WANDB_ENTITY needs to be set as an environment variable if not running in offline mode. Check .env file and source it if necessary."

    # Disable parallelism in tokenizers to avoid issues with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_config(cfg: TransformerSegmentationConfig) -> None:
    """Infer missing config values (checkpoint path and/or experiment name) and check if keys are missing"""

    # Infer resume_run_id if checkpoint path is provided
    if cfg.experiment.resume_checkpoint_path and not cfg.experiment.resume_run_id:
        if "name" not in cfg.experiment:
            cfg.experiment.name = cfg.experiment.resume_checkpoint_path.split("/")[-2]
            logger.warning(f"experiment.name not set, infering {cfg.experiment.name} from resume_checkpoint_path.")
        wandb_entity = os.environ.get("WANDB_ENTITY")
        api = wandb.Api()
        runs = api.runs(f"{wandb_entity}/{cfg.experiment.group}")
        for run in runs:
            if run.name == cfg.experiment.name:
                cfg.experiment.resume_run_id = run.id
                logger.info(f"resume_run_id not set, loaded {cfg.experiment.resume_run_id} from resume_checkpoint_path.")
                break

    # It is possible to infer the name if resume_run_id is set or if resume_checkpoint_path is set. Otherwise, a random name is generated.
    if cfg.experiment.resume_run_id:
        # Case when resume_run_id provided but not the experiment name
        wandb_entity = os.environ.get("WANDB_ENTITY")
        if "name" not in cfg.experiment:
            api = wandb.Api()
            run = api.run(f"{wandb_entity}/{cfg.experiment.group}/{cfg.experiment.resume_run_id}")
            cfg.experiment.name = run.name
            logger.info(
                f"experiment.name not set, loaded {cfg.experiment.name} from resume_run_id {cfg.experiment.resume_run_id} on wandb."
            )
        # Case when resume_run_id provided but not the checkpoint path
        if not cfg.experiment.resume_checkpoint_path:
            checkpoint_paths = [
                dir for dir in os.listdir(f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}") if dir.startswith("checkpoint")
            ]
            if len(checkpoint_paths) > 0:
                checkpoint_numbers = [int(path.split("-")[-1]) for path in checkpoint_paths]
                checkpoint_numbers.sort()
                cfg.experiment.resume_checkpoint_path = (
                    f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}/checkpoint-{checkpoint_numbers[-1]}"
                )
                logger.info(f"resume_checkpoint_path not set, loaded {cfg.experiment.resume_checkpoint_path} from latest checkpoint.")
            else:
                raise RuntimeError(
                    f"resume_run_id set but no checkpoints found in the run directory checkpoints/{cfg.experiment.group}/{cfg.experiment.name}. Please specify resume_checkpoint_path."
                )
    if "name" not in cfg.experiment:
        # Case when checkpoint_path is provided but not the experiment name
        if cfg.experiment.resume_checkpoint_path is not None:
            cfg.experiment.name = cfg.experiment.resume_checkpoint_path.split("/")[-2]
            logger.warning(f"experiment.name not set, infering {cfg.experiment.name} from resume_checkpoint_path.")
        # Case when neither resume_run_id nor resume_checkpoint_path is provided. Generate a random name.
        else:
            cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(9999, (1,)).item()).zfill(4)}"
            if not cfg.experiment.offline_run:
                wandb_entity = os.environ.get("WANDB_ENTITY")
                api = wandb.Api()
                try:
                    runs = api.runs(f"{wandb_entity}/{cfg.experiment.group}")
                    while any(run.name == cfg.experiment.name for run in runs):
                        cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(9999, (1,)).item()).zfill(4)}"
                except Exception as e:
                    logger.warning(f"Failed to check if experiment name {cfg.experiment.name} is unique on wandb. Error: {e}")
                    cfg.experiment.name = f"{cfg.dataset.subconfig}-{str(torch.randint(9999, (1,)).item()).zfill(4)}"
            logger.warning(f"experiment.name not set, generated random name {cfg.experiment.name}")

    # Raise error if any keys are missing
    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")
    if cfg.data_preprocessing.join_utts not in ["dynamic", "static", None, "None"]:
        raise RuntimeError(f"Invalid value for join_utts: {cfg.data_preprocessing.join_utts}. Must be one of 'dynamic', 'static', or None.")
    if cfg.data_preprocessing.subsample_type not in ["examples", "words", "tokens", None]:
        raise RuntimeError(
            f"Invalid value for subsample_type: {cfg.data_preprocessing.subsample_type}. Must be one of 'examples', 'words', or 'tokens'."
        )

    # Fix values
    if cfg.data_preprocessing.join_utts == "None":
        cfg.data_preprocessing.join_utts = None

    # Trainer step defaults
    if cfg.trainer.logging_steps is None:
        cfg.trainer.logging_steps = cfg.trainer.max_training_steps // 100  # log every 1% of training
    if cfg.trainer.eval_steps is None:
        cfg.trainer.eval_steps = cfg.trainer.max_training_steps // 10  # evaluate every 10% of training
    if cfg.trainer.save_steps is None:
        cfg.trainer.save_steps = cfg.trainer.max_training_steps // 10


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TransformerSegmentationConfig):
    check_and_set_environment_variables(cfg)
    check_config(cfg)
    setup.set_seed(cfg.experiment.seed)
    if cfg.experiment.dry_run:
        logger.info("Running in dry run mode -- overriding config with values: ")
        logger.info(f"\t max_training_steps: {DRY_RUN_TRAIN_STEPS}")
        logger.info(f"\t num_warmup_steps: {DRY_RUN_WARMUP_STEPS}")
        logger.info(f"\t eval_steps: {DRY_RUN_WARMUP_STEPS // 2}")
        logger.info(f"\t save_steps: {DRY_RUN_WARMUP_STEPS // 2}")
        logger.info(f"\t logging_steps: {DRY_RUN_WARMUP_STEPS // 100}")
        cfg.trainer.max_training_steps = DRY_RUN_TRAIN_STEPS
        cfg.trainer.num_warmup_steps = DRY_RUN_WARMUP_STEPS
        cfg.trainer.eval_steps = DRY_RUN_TRAIN_STEPS // 2
        cfg.trainer.save_steps = DRY_RUN_TRAIN_STEPS // 2
        cfg.trainer.logging_steps = DRY_RUN_TRAIN_STEPS // 100
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    logger.info("Loading dataset")
    dataset = setup.load_dataset(cfg.dataset)
    logging.info(f"Dataset loaded with {len(dataset['train'])} training examples")
    if cfg.experiment.dry_run:
        logger.info(f"Running in dry run mode -- subsampling dataset by {DRY_RUN_SUBSAMPLE_FACTOR}x")
        dataset["train"] = dataset["train"].select(range(0, dataset["train"].num_rows, DRY_RUN_SUBSAMPLE_FACTOR))
        dataset["valid"] = dataset["valid"].select(range(0, dataset["valid"].num_rows, DRY_RUN_SUBSAMPLE_FACTOR))
        cfg.experiment.segmentation_subsample = cfg.experiment.segmentation_subsample // DRY_RUN_SUBSAMPLE_FACTOR

    logger.info("Loading tokenizer")
    tokenizer = setup.load_tokenizer(cfg.tokenizer)

    if cfg.data_preprocessing.join_utts == "dynamic":
        # Set up custom data collator which joins examples to fill the context size
        data_collator = CustomDataCollatorForLanguageModeling(tokenizer, max_seq_length=cfg.data_preprocessing.max_input_length, mlm=False)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    logger.info("Initializing model")
    model = load_model(cfg.model, tokenizer)

    # Subsample words before tokenization as this may remove word boundaries
    if cfg.data_preprocessing.subsample is not None and cfg.data_preprocessing.subsample_type == "words":
        logger.info(f"Subsampling dataset to {cfg.data_preprocessing.subsample} {cfg.data_preprocessing.subsample_type}")
        dataset["train"] = setup.subsample_dataset_pre_tokenized(
            dataset["train"], cfg.data_preprocessing.subsample, cfg.data_preprocessing.subsample_type
        )

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DataPreprocessor(cfg.data_preprocessing, tokenizer, get_word_boundaries=cfg.experiment.evaluate_segmentation)
    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=(8 if torch.cuda.is_available() else 1),
        remove_columns=["text"],
    )
    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["valid"]

    # Subsample training dataset for other subsampling types
    if cfg.data_preprocessing.subsample is not None and cfg.data_preprocessing.subsample_type != "words":
        logger.info(f"Subsampling dataset to {cfg.data_preprocessing.subsample} {cfg.data_preprocessing.subsample_type}")
        train_dataset = setup.subsample_dataset(train_dataset, cfg.data_preprocessing.subsample, cfg.data_preprocessing.subsample_type)

    # Report key metrics
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")
    logger.info(f"Number of training examples: {train_dataset.num_rows}")
    logger.info(f"Number of validation examples: {eval_dataset.num_rows}")

    # Setting up wandb
    if not cfg.experiment.offline_run:
        setup.setup_wandb(cfg.experiment)
        wandb.config.update({"num_params": num_params})
        wandb.config.update({"num_train_examples": train_dataset.num_rows})
        wandb.config.update({"num_val_examples": eval_dataset.num_rows})

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        eval_strategy="steps",
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        per_device_eval_batch_size=cfg.trainer.batch_size,
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        eval_steps=cfg.trainer.eval_steps,
        save_steps=cfg.trainer.save_steps,
        logging_steps=cfg.trainer.logging_steps,
        run_name=cfg.experiment.name,
        report_to="wandb" if not cfg.experiment.offline_run else None,  # wandb deactivated for dry runs
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=not cfg.experiment.offline_run,
        hub_model_id=f"phonemetransformers/{cfg.experiment.group}-{cfg.model.name}-model" if not cfg.experiment.offline_run else None,
        hub_token=os.environ["HF_WRITE_TOKEN"] if not cfg.experiment.offline_run else None,
        remove_unused_columns=True,
        label_names=["input_ids"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,  # smaller perplexity is better
        use_cpu=not torch.cuda.is_available(),
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        is_phonemes=cfg.dataset.is_phonemes,
    )

    # Initial model evaluation
    if not cfg.experiment.resume_checkpoint_path:
        trainer.evaluate()

    # If resuming from final checkpoint, re-evaluate the final model and don't train
    if (
        cfg.experiment.resume_checkpoint_path
        and int(cfg.experiment.resume_checkpoint_path.split("/")[-1].split("-")[-1]) == cfg.trainer.max_training_steps
    ):
        trainer._load_from_checkpoint(cfg.experiment.resume_checkpoint_path)
        trainer.state = TrainerState.load_from_json(os.path.join(cfg.experiment.resume_checkpoint_path, "trainer_state.json"))
        trainer.evaluate()
        trainer._load_best_model()
        trainer.stride_evaluation = 2
        trainer.evaluate(metric_key_prefix="eval_best")

    # Train model
    else:
        trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

        # Evaluate best model
        trainer.stride_evaluation = 2
        trainer.evaluate(metric_key_prefix="eval_best")

        # Push final model with tag
        if not cfg.experiment.offline_run:
            trainer.push_to_hub(
                commit_message=f"Final model for experiment {cfg.experiment.name}",
                tags=cfg.experiment.name,  # Can be a string or list of strings
            )


if __name__ == "__main__":
    main()
