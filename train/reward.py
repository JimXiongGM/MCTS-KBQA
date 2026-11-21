import random
from glob import glob
from typing import Dict, List

import torch
import transformers

from common.common_utils import read_json
from common.training import log_rank_0
from data_loader.load import check_messages_format, make_sft_data_for_dialog
from data_loader.template import auto_detect_template

# Step-level reward training uses the original project's short scoring prompt.
REWARD_PROMPT_STEP_SHORT = """You are an expert evaluator. Your task is to assess the quality of an agent's reasoning for a knowledge graph (KG) question answering task. Score the current state of the agent from 0 to 100.

The agent can interact with a knowledge graph using these tools:

1. SearchNodes(query)
   - Searches for nodes in the knowledge graph by name

2. ExecuteSPARQL(sparql)
   - Executes SPARQL queries to explore the knowledge graph

3. SearchGraphPatterns(sparql, semantic)
   - Queries subgraphs where parameter 'sparql' must start with "SELECT ?e WHERE"
   - Returns subgraphs with ?e as head or tail entities
   - Uses 'semantic' parameter to sort results by predicate meaning
   - Note: For multi-value predicates, returns one random value
   - Note: Handles Freebase's CVT (Compound Value Type) nodes by combining two-hop relationships into one

The agent has interacted with the KG and you need to score the intermediate steps.

Please format your response as:
### Evaluation Reason: [Your detailed assessment]
### Score: [0-100]

Now, please evaluate the reasoning process according to the requirements. Be concise and clear.
"""


class StepRewardDataset(torch.utils.data.Dataset):
    """
    Given a question and the history of the dialog, predict the score and the reasoning process.
    format:
    ### Evaluation Reason: xxx
    ### Score: xxx
    """

    def __init__(
        self,
        raw_data: List[Dict],
        tokenizer: transformers.PreTrainedTokenizer,
        debug: bool = False,
    ):
        super(StepRewardDataset, self).__init__()
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

        self._load_data()

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _load_data(self):
        """
        each data d:
            - d["input_msg"]: list[dict]
            - d["eval_reason"]: str
            - d["score"]: float
        """
        all_inps, all_outs = [], []
        for d in self.raw_data:
            messages = d["input_msg"]
            assert (
                messages[0]["role"] == "system"
            ), f"the first role must be system, but got {messages[0]['role']}"

            # Flatten the existing conversation into input for the scoring model.
            turns = [messages[1]["content"]]
            for message in messages[2:]:
                prefix = "Assistant: " if message["role"] == "assistant" else ""
                if message["role"] in {"assistant", "user"}:
                    turns.append(prefix + message["content"])
            m_text = "\n".join(turns)

            # output
            # out_text = f"### Evaluation Reason: {d['eval_reason']}\n### Score: {int(d['score'])}"
            out_text = d["output"]

            new_messages = [
                {"role": "system", "content": REWARD_PROMPT_STEP_SHORT},
                {"role": "user", "content": m_text},
                {"role": "assistant", "content": out_text},
            ]

            check_messages_format(new_messages)

            inp_ids, target_mask = make_sft_data_for_dialog(
                new_messages, self.tokenizer, self.template
            )

            inp_ids = torch.LongTensor(inp_ids)
            target_mask = torch.BoolTensor(target_mask)
            out_ids = torch.where(
                target_mask, inp_ids, torch.tensor(self.tokenizer.pad_token_id)
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
                in_text = self.tokenizer.decode(
                    [t if t != self.tokenizer.pad_token_id else 0 for t in all_inps[_i]]
                )
                out_text = self.tokenizer.decode(
                    [t if t != self.tokenizer.pad_token_id else 0 for t in all_outs[_i]]
                )
                print(in_text)
                print("-" * 100)
                print(out_text)
                print("-" * 100)

            # lengths of inp_ids, out_ids
            inp_lens = [len(i) for i in all_inps]

            from evaluation.utils import statistic_info

            s = statistic_info(inp_lens)
            print("inp_lens:", s)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        _n = (self.labels[i] != self.tokenizer.pad_token_id).sum().item()
        percent = 100 * (self._No % self.__len__()) / self.__len__()
        log_rank_0(
            f"Global No. {self._No} index {i} ({percent:.2f}%) input_ids len: {len(self.input_ids[i])} labels len: {_n}",
            level="info",
        )
        self._No += self._world_size
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i].tolist(),
            attention_mask=self.input_ids[i].ne(self.tokenizer.pad_token_id),
        )


def make_dataset(
    dataset_patterns: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    debug=False,
):
    """
    dataset_patterns: "save-anno-reward/cwq/*.json"
    """
    paths = []
    for pattern in dataset_patterns:
        _paths = glob(pattern)
        assert len(_paths) > 0, f"No files found for pattern: {pattern}"
        log_rank_0(f"Find {len(_paths)} files for pattern: {pattern}")
        paths.extend(_paths)

    log_rank_0(f"Find {len(paths)} files in total.")

    data = [read_json(p) for p in paths]

    train_dataset = StepRewardDataset(
        raw_data=data,
        tokenizer=tokenizer,
        debug=debug,
    )
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        padding="longest",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id,
    )

    return train_dataset, data_collator


if __name__ == "__main__":
    """
    python data_loader/reward.py --dataset_patterns "save-anno-reward-v2/webqsp/*.json"
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
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
        type=str,
        default=["save-anno-reward-v2/webqsp/*.json"],  # webqsp, cwq
        help="reward anno data path pattern",
    )

    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        use_fast=False,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 4096

    train_dataset, data_collator = make_dataset(
        args.dataset_patterns, tokenizer, debug=True
    )
    x = 1
