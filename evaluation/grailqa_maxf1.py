import os
from glob import glob

from loguru import logger

from common.common_utils import underline_text
from evaluation.runtime import (
    _clean_float,
    _clean_latin_encoding,
    clean_xsd,
    extract_dialog_last_valid_observation,
)
from evaluation.webqsp import CalculatePRF1
from search.base import KBQATreeSearch


def _resolve_pkl_dir(save_dir: str):
    save_dir = save_dir.rstrip("/")
    if os.path.basename(save_dir) == "pkl" and os.path.isdir(save_dir):
        return save_dir

    pkl_dir = os.path.join(save_dir, "pkl")
    if os.path.isdir(pkl_dir):
        return pkl_dir

    raise FileNotFoundError(f"no pkl dir in {save_dir}")


def _max_f1_of_item(mcts):
    golden_answers = _clean_latin_encoding(clean_xsd(mcts.golden_answer))
    f1s = []
    for done_path in mcts.get_valid_done_paths():
        messages = mcts.get_messages_from_path(done_path)
        prediction = extract_dialog_last_valid_observation({"messages": messages})
        prediction = _clean_float(prediction)
        prediction = _clean_latin_encoding(clean_xsd(prediction))
        f1s.append(100 * CalculatePRF1(golden_answers, prediction)[2])
    max_f1 = max(f1s) if f1s else 0.0
    return round(max_f1, 2), bool(abs(max_f1 - 0) < 1e-6)


def _load_items(pkl_dir: str):
    items = []
    idx = 1
    for path in glob(os.path.join(pkl_dir, "*.pkl")):
        try:
            mcts = KBQATreeSearch.load(path)
            max_f1, all_empty = _max_f1_of_item(mcts)
            items.append(
                {
                    "id": str(mcts.qid),
                    "qtype": getattr(mcts, "qtype", "unknown"),
                    "max_f1": max_f1,
                    "all_empty": all_empty,
                }
            )
            if all_empty:
                print(f"No {idx}. {mcts.qid}")
                idx += 1
        except Exception as e:
            logger.warning(f"Error reading pkl {path}: {e}")
    return items


def _summarize(items):
    if not items:
        return 0.0, 0.0
    max_f1s = [item["max_f1"] for item in items]
    all_empty_flags = [item["all_empty"] for item in items]
    max_f1 = round(sum(max_f1s) / len(max_f1s), 2)
    all_empty = round(100 * sum(all_empty_flags) / len(all_empty_flags), 2)
    return max_f1, all_empty


def run(save_dir):
    pkl_dir = _resolve_pkl_dir(save_dir)
    items = _load_items(pkl_dir)

    iid_items = [item for item in items if item["qtype"] == "i.i.d."]
    comp_items = [item for item in items if item["qtype"] == "compositional"]
    zs_items = [item for item in items if item["qtype"] == "zero-shot"]
    total_items = items

    print(f"save_dir: {underline_text(save_dir)}")
    print(
        f"i.i.d.: {len(iid_items)}\tcompositional: {len(comp_items)}\tzero-shot: {len(zs_items)}\ttotal: {len(total_items)}"
    )
    print(f"{'type':<13}\t{'MaxF1@k':>5}\t{'Empty%@k':>9}")

    for name, subset in [
        ("i.i.d.", iid_items),
        ("compositional", comp_items),
        ("zero-shot", zs_items),
        ("total", total_items),
    ]:
        max_f1, all_empty = _summarize(subset)
        print(f"{name:<13}\t{max_f1:>5.2f}\t{all_empty:>9.2f}")


if __name__ == "__main__":
    run(save_dir="save-mcts/Actor-Meta-Llama-3.1-8B-Instruct-epoch5-v1/grailqa/pkl/")
