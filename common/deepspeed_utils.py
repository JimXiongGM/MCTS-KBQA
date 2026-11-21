import os
import random

import deepspeed
import numpy as np
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import set_seed


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = zero_stage == 3
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = (
                _z3_params_to_fetch([param, param_ema]) if zero_stage_3 else []
            )
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                params_to_fetch, enabled=should_gather_param
            ):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir):
    """
    Attention:
        model.layers.xxx.self_attn.rotary_emb.inv_freq are all missing
    note: model.state_dict() != model.named_parameters()
    model.state_dict() v: torch.Tensor
    model.named_parameters() v: torch.nn.parameter.Parameter
    """
    os.makedirs(save_dir, exist_ok=True)
    output_model_file = os.path.join(save_dir, "pytorch_model.bin")

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema

    output_state_dict = {}
    # fix
    for k1, v1 in model_to_save.state_dict().items():
        if k1.endswith(".inv_freq"):
            output_state_dict[k1] = v1.cpu()

    for k, v in model_to_save.named_parameters():
        if hasattr(v, "ds_id"):
            with deepspeed.zero.GatheredParameters(
                _z3_params_to_fetch([v]), enabled=True
            ):
                v_p = v.data.cpu()
        else:
            v_p = v.cpu()
        if global_rank == 0 and "lora" not in k:
            output_state_dict[k] = v_p

    # sorted
    output_state_dict = {
        k: output_state_dict[k] for k in sorted(output_state_dict.keys())
    }

    if global_rank == 0:
        torch.save(output_state_dict, output_model_file)
    del output_state_dict


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def save_hf_format(model, tokenizer, save_dir):
    """Save the model and tokenizer configuration needed to load the weights for inference."""
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.config.save_pretrained(save_dir)
    if model_to_save.generation_config is not None:
        model_to_save.generation_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def save_optimizer(optimizer, lr_scheduler, save_dir):
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    torch.save(checkpoint, os.path.join(save_dir, f"optimizer_scheduler.pt"))


# Function to load model, optimizer, and scheduler states
def load_optimizer(optimizer, lr_scheduler, save_dir):
    checkpoint = torch.load(os.path.join(save_dir, f"optimizer_scheduler.pt"))
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    return checkpoint["epoch"]


"""
if args.resume_training:
    last_epoch = load_checkpoint(args.resume_epoch, model, optimizer, lr_scheduler, args.output_dir)
    log_rank_0(f"Resumed from epoch {last_epoch}")
"""
