from dataclasses import dataclass
from typing import Dict

from common.training import log_rank_0


@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    system: str
    stop_word: str


template_dict: Dict[str, Template] = dict()


def register_template(
    template_name, system_format, user_format, assistant_format, system, stop_word=None
):
    """Register one chat template for SFT tokenization."""
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        system=system,
        stop_word=stop_word,
    )


register_template(
    template_name="llama3",
    system_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_format="{content}<|eot_id|>",
    system=None,
    stop_word="<|eot_id|>",
)

register_template(
    template_name="llama2",
    system_format="<<SYS>>\n{content}\n<</SYS>>\n\n",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content} </s>",
    system="You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.",
    stop_word="</s>",
)

register_template(
    template_name="qwen",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system="You are a helpful assistant.",
    stop_word="<|im_end|>",
)


def auto_detect_template(model_name):
    """Pick the Llama or Qwen chat template from the model name."""
    if "Llama-3" in model_name:
        name = "llama3"
    elif "Llama-2" in model_name:
        name = "llama2"
    elif "Qwen" in model_name or "qwen" in model_name:
        name = "qwen"
    else:
        raise ValueError(f"No chat template for model: {model_name}")
    log_rank_0(f"Auto detect template: {name}")
    return template_dict[name]
