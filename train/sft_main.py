"""
adapted from: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
"""

import json
import math
import os
import shutil
import time

import deepspeed
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.integrations import HfDeepSpeedConfig

from common.common_utils import time_diff_to_str
from common.deepspeed_utils import (
    get_optimizer_grouped_parameters,
    save_hf_format,
    set_random_seed,
    to_device,
)
from common.training import (
    get_system_info,
    log_rank_0,
    print_trainable_parameters,
    save_log_to_dir,
)
from train.sft_args import parse_args


def load_tokenizer(model_name_or_path):
    """Load the tokenizer from the local model directory."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
        padding_side="right",
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        log_rank_0(
            f"Set pad token to eos token: {tokenizer.eos_token}", level="warning"
        )
    tokenizer.model_max_length = 8000
    return tokenizer


def load_model(model_name_or_path, training=True, debug=False, dtype=None):
    """Initialize the model for training or inference from local weights."""
    config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        local_files_only=True,
    )
    if training:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=dtype or torch.float32,
            local_files_only=True,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map="auto",
            use_cache=True,
        )
        model.eval()

        if torch.__version__ >= "2.0.0":
            model = torch.compile(model)

    return model


def cosine_warmup_lr_scale(
    current_step, num_warmup_steps, num_training_steps, min_lr_rate=0.0, num_cycles=0.5
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0.0, factor)


def set_optimizer_learning_rate(
    optimizer,
    base_lrs,
    current_step,
    num_warmup_steps,
    num_training_steps,
    min_lr_rate=0.0,
):
    lr_scale = cosine_warmup_lr_scale(
        current_step=current_step,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_rate=min_lr_rate,
    )
    last_lr = []
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        param_group["lr"] = base_lr * lr_scale
        last_lr.append(param_group["lr"])
    return last_lr


def train(debug=False):
    if args.mode == "logic_form":
        from train.lf_sft import make_dataset
    elif args.mode == "interactive":
        from train.interactive_sft import make_dataset
    elif args.mode == "reward":
        from train.reward import make_dataset
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ds init
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # device_id avoids NCCL "Guessing device ID" warning from barrier()
        torch.distributed.init_process_group(backend="nccl", device_id=device)
        deepspeed.init_distributed()
        log_rank_0("deepspeed init done")

    log_rank_0(get_system_info())
    # resume.
    if args.resume_dir:
        resume_dir = os.path.join(args.resume_dir, "optimizer_scheduler.pt")
        assert os.path.exists(resume_dir), f"resume_dir not exists: {resume_dir}"

    set_random_seed(args.seed)
    torch.distributed.barrier()

    # Load tokenizer
    log_rank_0("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name_or_path)
    log_rank_0(f"tokenizer.model_max_length: {tokenizer.model_max_length}")

    # Load data
    log_rank_0("Loading data...")
    train_dataset, data_collator = make_dataset(
        args.dataset_patterns, tokenizer=tokenizer, debug=debug
    )

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=False)

    with open(args.ds_config_file) as config_file:
        ds_config = json.load(config_file)
    # Split the global batch size across ranks and gradient accumulation steps.
    per_device_train_batch_size = args.per_device_train_batch_size
    world_size = torch.distributed.get_world_size()
    micro_batch_size = per_device_train_batch_size * world_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps, remainder = divmod(
            args.total_batch_size, micro_batch_size
        )
        if remainder or gradient_accumulation_steps < 1:
            raise ValueError(
                f"Total batch size must be a positive multiple of {micro_batch_size}."
            )
    if gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive.")
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    ds_config["train_batch_size"] = micro_batch_size * gradient_accumulation_steps

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        sampler=train_sampler,
        batch_size=per_device_train_batch_size,
    )
    torch.distributed.barrier()

    log_rank_0(
        f"gradient_accumulation_steps: {gradient_accumulation_steps} "
        f"train_batch_size: {ds_config['train_batch_size']}"
    )

    # Enable ZeRO-3 sharding before loading weights and keep it active while the model loads.
    zero_config = ds_config.get("zero_optimization", {})
    args.zero_stage = zero_config.get("stage", 0)
    if args.zero_stage == 3:
        zero_config["stage3_gather_16bit_weights_on_model_save"] = True
    args.ds_config = ds_config
    hf_ds_config = HfDeepSpeedConfig(ds_config)
    log_rank_0("Loading model...")
    model = load_model(args.model_name_or_path, debug=debug)
    del hf_ds_config

    # optimizer
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # lr_scheduler
    total_scheduler_steps = (
        len(train_dataloader) / gradient_accumulation_steps * args.num_train_epochs
    )
    total_scheduler_steps = math.ceil(total_scheduler_steps)

    if "scheduler" in ds_config or "optimizer" in ds_config:
        raise ValueError("We use scheduler and optimizer from transformers")

    num_warmup_steps = math.ceil(total_scheduler_steps * args.warmup_ratio)
    min_lr_rate = 0.05
    log_rank_0(
        f"num_training_steps (total_scheduler_steps): {total_scheduler_steps}, num_warmup_steps: {num_warmup_steps}"
    )
    log_rank_0(print_trainable_parameters(model, str_mode=True))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        dist_init_required=True,
    )

    base_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
    last_lrs = set_optimizer_learning_rate(
        optimizer=optimizer,
        base_lrs=base_lrs,
        current_step=0,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_scheduler_steps,
        min_lr_rate=min_lr_rate,
    )

    total_steps = len(train_dataloader) * args.num_train_epochs
    global_step = 0
    scheduler_step = 0
    log_rank_0(f"total_steps: {total_steps}", level="warning")
    save_dir = None
    torch.distributed.barrier()
    start_time = time.time()
    for epoch in range(args.num_train_epochs):
        log_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches: {len(train_dataloader)}",
            level="warning",
        )
        for step, batch in enumerate(train_dataloader):
            # DeepSpeed requires a step call for each microbatch. The learning rate updates on optimizer steps.
            update = (
                global_step + 1
            ) % gradient_accumulation_steps == 0 or global_step + 1 == total_steps
            model.set_gradient_accumulation_boundary(update)
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()

            if update:
                scheduler_step += 1
                last_lrs = set_optimizer_learning_rate(
                    optimizer=optimizer,
                    base_lrs=base_lrs,
                    current_step=scheduler_step,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_scheduler_steps,
                    min_lr_rate=min_lr_rate,
                )
                current_lr = last_lrs[0]
                persent_done = scheduler_step / total_scheduler_steps

                current_time = time.time()
                time_cost_fmt = time_diff_to_str(current_time - start_time)
                time_cost_per_percent = (current_time - start_time) / persent_done
                time_remain_fmt = time_diff_to_str(
                    time_cost_per_percent * (1 - persent_done)
                )

                log_rank_0(
                    f"Global Step: {global_step+1} Epoch: {epoch+1} Step: {step+1} "
                    f"Global Scheduler Step: {scheduler_step+1} ({100*persent_done:.2f}%) "
                    f"Loss: {loss.item():.8f} Learning Rate: {current_lr}",
                    level="debug",
                )
                log_rank_0(
                    f"Time Cost: {time_cost_fmt} Time Remain: {time_remain_fmt}",
                    level="debug",
                )

            global_step += 1

        model.tput_timer.update_epoch_count()

        # Export an SGLang-compatible model directory after the final epoch.
        if epoch + 1 != args.num_train_epochs:
            continue

        if args.output_dir is not None:
            save_dir = args.output_dir + f"-epoch{epoch + 1}"
            log_rank_0(f"Saving model at epoch {epoch + 1} to {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            if torch.distributed.get_rank() == 0:
                if args.zero_stage == 3:
                    save_hf_format(model, tokenizer, save_dir)
                else:
                    model.module.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4, ensure_ascii=False)
                save_log_to_dir(save_dir)
                shutil.copy2(__file__, save_dir)
            if args.zero_stage == 3 and not model.save_16bit_model(save_dir):
                raise RuntimeError(f"Could not export model weights to {save_dir}")

    log_rank_0(f"Done! Saved to: {save_dir}")


if __name__ == "__main__":
    log_rank_0(file_name=os.path.basename(__file__))
    args = parse_args()
    train(debug=0)
