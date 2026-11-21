import random
from glob import glob
from typing import Dict, List

import torch
import transformers

from common.common_utils import read_json
from common.training import LABEL_PAD_TOKEN_ID, log_rank_0
from train.interactive_sft import (
    ConversationDatasetForCasualLM,
    make_sft_data_for_dialog,
)

LOGIC_FORM_INSTRUCTION = """You are a helpful assistant. Please write a SPARQL query to answer the question."""


class SFTDataset(ConversationDatasetForCasualLM):
    """
    Given a question, predict the Logical Form (LF) of the question.
    instruction:
        - xxx
        - xxx.\nQ:xxx\nSPARQL:xxx
    """

    def __init__(self, *args, **kwargs):
        self.instruction = LOGIC_FORM_INSTRUCTION
        super().__init__(*args, **kwargs)

    def _flatten_turns(self):
        """
        {"role": "system", "content": "{instruction}"}
        {"role": "user", "content": "Q: xxx"}
        {"role": "assistant", "content": "SPARQL: xxx"}

        label: SPARQL: xxx
        """
        all_inps, all_outs = [], []
        for d in self.raw_data:
            messages = [{"role": "system", "content": self.instruction}]
            messages.append({"role": "user", "content": "Q: " + d["question"]})
            messages.append({"role": "assistant", "content": "SPARQL: " + d["sparql"]})
            inp_ids, target_mask = make_sft_data_for_dialog(
                messages, self.tokenizer, self.template
            )

            inp_ids = torch.LongTensor(inp_ids)
            target_mask = torch.BoolTensor(target_mask)
            out_ids = torch.where(
                target_mask, inp_ids, torch.tensor(LABEL_PAD_TOKEN_ID)
            )

            all_inps.append(inp_ids)
            all_outs.append(out_ids)

        # shuffle
        random.seed(42)
        tmp = list(zip(all_inps, all_outs))
        random.shuffle(tmp)
        all_inps, all_outs = zip(*tmp)

        self.input_ids = all_inps
        self.labels = all_outs
        log_rank_0(f"len(self.input_ids): {len(self.input_ids)}")

        if self.debug:
            for _i in range(min(10, len(all_outs))):
                x = self.tokenizer.decode(
                    [t if t != LABEL_PAD_TOKEN_ID else 0 for t in all_outs[_i]]
                )
                print(x)
                print()

            # lengths of inp_ids, out_ids
            inp_lens = [len(i) for i in all_inps]

            from evaluation.utils import statistic_info

            s = statistic_info(inp_lens)
            print("inp_lens:", s)


def make_dataset(dataset_patterns, tokenizer, debug=False):
    raw_data = []
    for patt in dataset_patterns:
        paths = glob(patt)
        assert len(paths) > 0, f"No json file found in {patt}"
        log_rank_0(f"Find paths: {len(paths)}")

        _raw_data = []
        for p in paths:
            item = read_json(p)
            if isinstance(item, dict):
                item = [item]
            elif isinstance(item, list):
                pass
            else:
                raise ValueError(f"Unknown item type: {type(item)}")
            _raw_data += item

        _raw_data = _raw_data[: 10 if debug else None]

        if "webqsp" in patt:
            dataset = "webqsp"
        elif "cwq" in patt:
            dataset = "cwq"
        elif "kqapro" in patt:
            dataset = "kqapro"
        else:
            raise ValueError(f"Unknown dataset: {patt}")

        for i in range(len(_raw_data)):
            _raw_data[i]["dataset"] = dataset

        raw_data += _raw_data

    train_dataset = SFTDataset(
        raw_data,
        tokenizer=tokenizer,
        debug=debug,
    )

    # logger.warning(f"pad token: {tokenizer.pad_token}")
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        # pad_to_multiple_of=4,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id,
    )
    return train_dataset, data_collator


if __name__ == "__main__":
    """
    python data_loader/lf_sft.py
    """
    import argparse
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="./meta-llama/Meta-Llama-3.1-8B-Instruct",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=5,
        type=int,
        help="Batch size per GPU/TPU core/CPU for training.",
    )
    parser.add_argument(
        "--dataset_patterns",
        nargs="+",
        default=["data/human-anno-v2/kqapro/*/*.json"],
        type=str,
        help="human anno data path pattern",
    )

    args = parser.parse_args()
    log_rank_0(file_name=os.path.basename(__file__))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        padding_side="right",
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log_rank_0(
            f"Set pad token to eos token: {tokenizer.eos_token}", level="warning"
        )
    tokenizer.model_max_length = 8000

    train_dataset, data_collator = make_dataset(
        args.dataset_patterns, tokenizer, debug=True
    )
    train_dataset.__getitem__(2)
    x = 1
