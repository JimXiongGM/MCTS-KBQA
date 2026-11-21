import argparse
import datetime
import os

import torch
from loguru import logger

LABEL_PAD_TOKEN_ID = -100

LOG_FILE = None


def log_rank_0(msg="", level="info", file_name=""):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    global LOG_FILE
    if LOG_FILE is None:
        os.makedirs("logs", exist_ok=True)
        LOG_FILE = f"logs/{file_name}-{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log"
        logger.add(LOG_FILE)

    if msg:
        if level == "info":
            logger.info(msg)
        elif level == "debug":
            logger.debug(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "error":
            logger.error(msg)
        elif level == "critical":
            logger.critical(msg)
        else:
            raise ValueError(f"level: {level} is not supported.")


def save_log_to_dir(dir_name):
    global LOG_FILE
    if LOG_FILE is None:
        return
    os.makedirs(dir_name, exist_ok=True)
    os.system(f"cp {LOG_FILE} {dir_name}/")


def parse_args():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--args_filename", type=str, default=None)
    args = _parser.parse_args()
    return args


def print_trainable_parameters(model, str_mode=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    _p = f"Parameters. all: {all_param}, trainable: {trainable_params} ({100 * trainable_params / all_param:.2f}%)"
    if str_mode:
        return _p
    else:
        print(_p)


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb

        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if "lm_head" in name:
                continue
            if "output_layer" in name:
                continue
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def get_system_info():
    """Get system information including OS, CPU, memory and GPU."""
    import subprocess

    import distro
    import psutil

    info = {}

    # OS info
    info["os"] = f"{distro.name()} {distro.version()}"

    # CPU info
    with open("/proc/cpuinfo") as f:
        cpu_info = f.read()
    cpu_name = [line for line in cpu_info.split("\n") if "model name" in line]
    if cpu_name:
        info["cpu"] = cpu_name[0].split(":")[1].strip()
    info["cpu_count"] = psutil.cpu_count()

    # Memory info
    memory = psutil.virtual_memory()
    info["memory_total"] = f"{memory.total / (1024**3):.1f}GB"
    info["memory_available"] = f"{memory.available / (1024**3):.1f}GB"

    # GPU info
    try:
        nvidia_smi = "nvidia-smi"
        result = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=gpu_name,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            encoding="utf-8",
        )
        gpu_info = []
        for line in result.strip().split("\n"):
            name, total, used = line.split(", ")
            gpu_info.append(f"{name} ({used}/{total})")
        info["gpu"] = gpu_info
    except:
        info["gpu"] = "No GPU found"

    return info


if __name__ == "__main__":
    print(get_system_info())
