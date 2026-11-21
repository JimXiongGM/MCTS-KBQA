import random
from glob import glob
from typing import Dict, List

import torch
import transformers

from common.common_utils import read_json
from common.training import LABEL_PAD_TOKEN_ID, log_rank_0
from data_loader.load import (
    check_messages_format,
    make_sft_data_for_dialog,
    make_system_prompt,
)
from data_loader.template import auto_detect_template


class ConversationDatasetForCasualLM(torch.utils.data.Dataset):

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        debug: bool = False,
    ):
        super().__init__()

        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.debug = debug
        self._No = 0
        self._world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

        self.template = auto_detect_template(tokenizer.name_or_path)

        self._flatten_turns()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        _n = (self.labels[i] != LABEL_PAD_TOKEN_ID).sum().item()
        percent = 100 * (self._No % self.__len__()) / self.__len__()
        log_rank_0(
            f"Global No. {self._No} index {i} ({percent:.2f}%) input_ids len: {len(self.input_ids[i])} labels len: {_n}",
            level="info",
        )
        self._No += self._world_size
        # Return labels as list so DataCollatorForSeq2Seq uses list path and avoids slow torch.tensor(list of ndarrays)
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i].tolist(),
            attention_mask=self.input_ids[i].ne(self.tokenizer.pad_token_id),
        )

    def _flatten_turns(self):
        """
        {"role": "system", "content": "You are an AI assistant."}
        {"role": "user", "content": "{prompt text}\nQ: xxx"}
        {"role": "assistant", "content": "xxx"}
        {"role": "user", "content": "Thought: xxx Action: xxx"}
        {"role": "assistant", "content": "Observation: xxx"}

        label: Thought: xxx\nAction: xxx
        """
        all_inps, all_outs = [], []
        for d in self.raw_data:
            # NOTE: Training and inference MUST be consistent.
            q = d["question"]
            system_prompt = make_system_prompt(d["dataset"], ft=True)

            messages = d["messages"] if "messages" in d else d["history"]
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages[0]["content"] = system_prompt

            check_messages_format(messages)
            messages[1]["content"] = f"Q: {q}"

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
        elif "grailqa" in patt:
            dataset = "grailqa"
        else:
            raise ValueError(f"Unknown dataset: {patt}")

        for i in range(len(_raw_data)):
            _raw_data[i]["dataset"] = dataset

        raw_data += _raw_data

    train_dataset = ConversationDatasetForCasualLM(
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
        label_pad_token_id=LABEL_PAD_TOKEN_ID,
    )
    return train_dataset, data_collator


if __name__ == "__main__":
    """
    python data_loader/interactive_sft.py
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
        default=["data/human-anno-v2/grailqa/*/*.json"],
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
    # train_dataset, data_collator = merge_all_dataset(args, tokenizer, debug=True)
    train_dataset.__getitem__(2)
    x = 1
