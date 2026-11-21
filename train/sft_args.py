import argparse

import deepspeed
from transformers import SchedulerType


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )

    # myadd
    parser.add_argument(
        "--ds_config_file",
        type=str,
        required=True,
        help="Path to a json configuration file to override args.",
    )
    parser.add_argument(
        "--dataset_patterns",
        nargs="+",
        type=str,
        required=True,
        help="webqsp: WebQSP; cwq: ComplexWebQuestions; kqapro: KQAPRO.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="logic_form, interactive",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to a local pretrained model directory.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=4096,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of updates steps to accumulate. If not set, derived from total_batch_size and per_device_train_batch_size * world_size.",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=8,
        help="Global effective batch size (per_device_train_batch_size * world_size * gradient_accumulation_steps).",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="add. Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )

    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )

    # Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step1_tensorboard")

    # resume training
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Directory to resume training from.",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args
