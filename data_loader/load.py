import random
from functools import lru_cache
from glob import glob
from typing import Dict, List

from loguru import logger
from transformers import PreTrainedTokenizer

from common.common_utils import read_json
from common.constant import TOOL_DESC_SHORT_FB, TOOL_DESC_SHORT_KQAPRO
from common.path_utils import (
    filter_by_sample_ids,
    get_dataset_split_glob,
    load_kqapro_test_data,
    resolve_dataset_base_dir,
)
from data_loader.template import Template


def normalize_dataset_item(d: dict, dataset: str) -> dict:
    """Align field names across dataset versions (e.g. GrailQA v3 uses sparql_query)."""
    if dataset == "grailqa" and "sparql" not in d:
        if "sparql_query" in d:
            d["sparql"] = d["sparql_query"]
        elif "sparql_query_readable" in d:
            d["sparql"] = d["sparql_query_readable"]
    if dataset == "kqapro":
        if "sparql" not in d and "sparql_query" in d:
            d["sparql"] = d["sparql_query"]
        if "qtype" not in d and "type" in d:
            d["qtype"] = d["type"]
    return d


def get_qtype_key(dataset: str) -> str:
    if dataset == "cwq":
        return "compositionality_type"
    elif dataset == "webqsp":
        return "infer_chain"
    elif dataset == "kqapro":
        return "qtype"
    elif dataset == "grailqa":
        return "level"
    else:
        raise ValueError(f"dataset: {dataset} not supported.")


def load_test_data(
    dataset,
    num_each_type=None,
    full=False,
    dataset_base_dir=None,
    use_sample=False,
):
    """Load evaluation questions, always using the fixed KQA Pro test set."""
    assert dataset in [
        "webqsp",
        "cwq",
        "kqapro",
        "grailqa",
    ], f"dataset: {dataset} not supported."
    # Read the fixed KQA Pro test set separately from the other datasets' file layouts.
    if dataset == "kqapro":
        data = load_kqapro_test_data(dataset_base_dir)[:num_each_type]
    else:
        split_glob = get_dataset_split_glob(dataset, full, dataset_base_dir)
        data = []
        for path in glob(split_glob):
            data.extend(read_json(path)[:num_each_type])
    assert data, f"No evaluation questions for {dataset}"
    data = filter_by_sample_ids(
        data, dataset, use_sample=use_sample, dataset_base_dir=dataset_base_dir
    )
    if use_sample and dataset in ("webqsp", "cwq"):
        logger.info(f"use_sample: loaded {len(data)} items for {dataset}")
    data = [normalize_dataset_item(d, dataset) for d in data]
    data.sort(key=lambda x: x["id"])
    return data


# d=load_test_data("webqsp");print(len(d))
# d=load_test_data("cwq");print(len(d))
# exit()


def load_train_data(dataset):
    assert dataset in ["webqsp", "cwq", "kqapro"], f"dataset: {dataset} not supported."
    p = f"data/human-anno-v2/{dataset}/*/*.json"
    paths = glob(p)
    data = [read_json(p) for p in paths]
    return data


def _sample_cwq_data_by_webqspid(data):
    """
    group by webqsp_id, and sample some data from each group.
    """
    data_by_webqspid = {}
    for d in data:
        webqsp_id = d["id"].split("_")[0]
        if webqsp_id not in data_by_webqspid:
            data_by_webqspid[webqsp_id] = []
        data_by_webqspid[webqsp_id].append(d)

    # sample 2 data from each group
    sampled_data = []
    for webqsp_id, d_list in data_by_webqspid.items():
        sampled_data.extend(d_list[:1])

    return sampled_data


def load_train_unseen_data(dataset, num_each_type=1.0):
    assert dataset in ["webqsp", "cwq", "kqapro"], f"dataset: {dataset} not supported."
    seen_ids = [d["id"] for d in load_train_data(dataset)]
    assert len(seen_ids) > 0, f"seen_ids is empty."
    seen_ids = set(seen_ids)
    dataset_base_dir = resolve_dataset_base_dir()
    paths = glob(f"{dataset_base_dir}/{dataset}/train/*.json")
    data = []
    random.seed(42)
    for p in paths:
        _data = read_json(p)
        _data = [i for i in _data if i["id"] not in seen_ids]
        random.shuffle(_data)
        if num_each_type <= 1.0:
            _len = int(len(_data) * num_each_type)
        else:
            _len = int(num_each_type)
        data += _data[:_len]
    random.shuffle(data)

    # skip composition type data
    # data = [i for i in data if i["compositionality_type"] in ["superlative", "comparative"]]
    return data


@lru_cache(maxsize=128)
def make_system_prompt(dataset, ft=False, qtype=None):
    """Build the system prompt for one dataset.

    GrailQA always uses fewshot_demo/grailqa.txt. WebQSP, CWQ, and KQA Pro only
    keep the short tool description used by fine-tuned models.
    """
    if dataset == "grailqa":
        # GrailQA keeps its tool description and example in this one file.
        return open("fewshot_demo/grailqa.txt").read().strip()

    if not ft:
        raise ValueError(
            f"Few-shot prompts for {dataset} are not shipped in this repo. "
            "Use a fine-tuned model, or take the prompts from the interactive-kbqa repo."
        )
    if qtype:
        raise ValueError("qtype must be None for a fine-tuned prompt.")

    if dataset in ["webqsp", "cwq", "webqsp+cwq"]:
        return TOOL_DESC_SHORT_FB
    if dataset == "kqapro":
        return TOOL_DESC_SHORT_KQAPRO
    raise ValueError(f"dataset: {dataset} not supported.")


def check_messages_format(messages: List[Dict]):
    assert isinstance(messages[0], dict), "messages[0] must be dict"
    assert messages[0]["role"] == "system", "messages[0] must be system"

    for i in range(1, len(messages)):
        if i % 2 == 1:
            assert (
                messages[i]["role"] == "user"
            ), f"messages[{i}] must be user. get: {messages[i]['role']}"
        elif i % 2 == 0:
            assert (
                messages[i]["role"] == "assistant"
            ), f"messages[{i}] must be assistant. get: {messages[i]['role']}"


def make_sft_data_for_dialog(
    messages, tokenizer: PreTrainedTokenizer, template: Template
):
    """
    from firefly
    format: {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
    """
    input_ids, target_mask = [], []

    if messages[0]["role"] == "system":
        system = messages[0]["content"].strip()
        system_text = template.system_format.format(content=system)
        input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        target_mask = [0] * len(input_ids)

    for i, conv in enumerate(messages):
        if conv["role"] == "user":
            user = conv["content"].strip()
            user_text = template.user_format.format(
                content=user, stop_token=tokenizer.eos_token
            )
            user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
            input_ids += user_tokens
            target_mask += [0] * len(user_tokens)
        elif conv["role"] == "assistant":
            assistant = conv["content"].strip()
            assistant_text = template.assistant_format.format(
                content=assistant, stop_token=tokenizer.eos_token
            )
            assistant_tokens = tokenizer.encode(
                assistant_text, add_special_tokens=False
            )
            input_ids += assistant_tokens
            target_mask += [1] * len(assistant_tokens)

        # user = conv['user'].strip()
        # assistant = conv['assistant'].strip()

        # user = template.user_format.format(content=user, stop_token=tokenizer.eos_token)
        # assistant = template.assistant_format.format(content=assistant, stop_token=tokenizer.eos_token)

        # input_tokens = tokenizer.encode(user, add_special_tokens=False)
        # output_tokens = tokenizer.encode(assistant, add_special_tokens=False)

        # input_ids += input_tokens + output_tokens
        # target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

    input_ids = input_ids[: tokenizer.model_max_length]
    target_mask = target_mask[: tokenizer.model_max_length]
    assert len(input_ids) == len(target_mask)

    return input_ids, target_mask
