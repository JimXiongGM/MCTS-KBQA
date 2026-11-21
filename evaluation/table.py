from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache, partial
from glob import glob

import fire

from common.common_utils import read_json
from common.path_utils import (
    V3_DATASET_DIRS,
    V3_DATASET_FILES,
    filter_by_sample_ids,
    load_kqapro_test_ids,
    resolve_dataset_base_dir,
)
from evaluation.runtime import (
    clean_xsd,
    extract_dialog_last_valid_observation,
    extract_dialog_last_valid_observation_kqa,
    whether_equal,
)
from evaluation.webqsp import cal_metrics

ACTOR_NAME = "Meta-Llama-3.1-8B-Instruct-epoch5"
DEFAULT_DATASET_BASE_DIR = None
DEFAULT_SEEDS = ("1", "10", "100", "1000", "10000")
KQAPRO_QTYPES = (
    "Count",
    "QueryAttr",
    "QueryAttrQualifier",
    "QueryName",
    "QueryRelation",
    "QueryRelationQualifier",
    "SelectAmong",
    "SelectBetween",
    "Verify",
)
KQAPRO_QTYPE_SHORT = {
    "Count": "Ct",
    "QueryAttr": "QA",
    "QueryAttrQualifier": "QAQ",
    "QueryName": "QN",
    "QueryRelation": "QR",
    "QueryRelationQualifier": "QRQ",
    "SelectAmong": "SA",
    "SelectBetween": "SB",
    "Verify": "Vfy",
}


@lru_cache(maxsize=8)
def _load_v3_infer_chain_map(
    dataset: str, dataset_base_dir: str | None = None
) -> dict[str, list]:
    base_dir = resolve_dataset_base_dir(dataset_base_dir)
    if dataset not in V3_DATASET_DIRS:
        return {}
    path = f"{base_dir}/{V3_DATASET_DIRS[dataset]}/{V3_DATASET_FILES[dataset]}"
    items = read_json(path)
    return {
        str(d["id"]): d["infer_chain"]
        for d in items
        if d.get("infer_chain") is not None
    }


def _normalize_golden_answers(item: dict) -> None:
    if "answers" not in item and "answer" in item:
        item["answers"] = item["answer"]


def _attach_infer_chain(
    data: list[dict], dataset: str, dataset_base_dir: str | None
) -> list[dict]:
    mapping = _load_v3_infer_chain_map(dataset, dataset_base_dir)
    for item in data:
        if item.get("infer_chain") is None:
            chain = mapping.get(str(item.get("id", "")))
            if chain is not None:
                item["infer_chain"] = chain
    return data


def _infer_chain_hop_len(item: dict) -> int | None:
    chain = item.get("infer_chain")
    if not chain:
        return None
    return len(chain)


def _default_num_workers() -> int:
    """Use all host CPUs by default."""
    return max(1, os.cpu_count() or 1)


def _extract_prediction(item: dict, dataset: str | None, save_dir: str) -> list:
    if "prediction" in item and not item.get("history") and not item.get("messages"):
        return item["prediction"]
    if dataset == "kqapro":
        return extract_dialog_last_valid_observation_kqa(item)
    return extract_dialog_last_valid_observation(item)


def _load_one_prediction(
    path: str, dataset: str | None, test_ids: frozenset[str] | None
) -> dict | None:
    """Load one prediction and check the question scope before parsing the answer."""
    try:
        item = read_json(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None
    if test_ids is not None and str(item.get("id", "")) not in test_ids:
        return None
    _normalize_golden_answers(item)
    item["prediction"] = _extract_prediction(item, dataset, path)
    return item


def _load_json_predictions(
    save_dir: str,
    dataset: str | None = None,
    only_sample: bool = False,
    dataset_base_dir: str | None = None,
    num_workers: int | None = None,
) -> list[dict]:
    """Load predictions and keep only questions in the evaluation set."""
    paths = sorted(glob(f"{save_dir}/*.json"))
    workers = (
        _default_num_workers() if num_workers is None else max(1, int(num_workers))
    )
    test_ids = (
        load_kqapro_test_ids(dataset_base_dir)
        if dataset == "kqapro" and paths
        else None
    )
    load_one = partial(_load_one_prediction, dataset=dataset, test_ids=test_ids)
    if workers == 1 or len(paths) <= 1:
        data = [item for item in map(load_one, paths) if item is not None]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            data = [item for item in pool.map(load_one, paths) if item is not None]
    if dataset and dataset != "kqapro" and only_sample and data:
        data = filter_by_sample_ids(
            data, dataset, use_sample=True, dataset_base_dir=dataset_base_dir
        )
    return data


def _fmt(value):
    return "-" if value is None else f"{value:.2f}"


def _format_aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width ASCII table: Method left-aligned, data columns right-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def border(sep_char: str, join_char: str) -> str:
        parts = [sep_char * (w + 2) for w in widths]
        return join_char.join(parts)

    top = "+" + border("-", "+") + "+"
    mid = "+" + border("-", "+") + "+"
    bot = "+" + border("-", "+") + "+"

    def fmt_row(cells: list[str]) -> str:
        out = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            if i == 0:
                out.append(f" {cell:<{w}} ")
            else:
                out.append(f" {cell:>{w}} ")
        return "|" + "|".join(out) + "|"

    lines = [top, fmt_row(headers), mid]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(bot)
    return "\n".join(lines)


def _webqsp_metrics(
    save_dir: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    num_workers: int | None = None,
) -> tuple[str, str, str, str, str]:
    data = _load_json_predictions(
        save_dir,
        dataset="webqsp",
        only_sample=only_sample,
        dataset_base_dir=dataset_base_dir,
        num_workers=num_workers,
    )
    data = _attach_infer_chain(data, "webqsp", dataset_base_dir)
    n = str(len(data))
    if not data:
        return "-", "-", "-", "-", n

    groups = [
        [d for d in data if _infer_chain_hop_len(d) == 1],
        [d for d in data if _infer_chain_hop_len(d) == 2],
        deepcopy(data),
    ]
    values = []
    for subset in groups:
        if not subset:
            values.append((None, None))
            continue
        golden_answers = [clean_xsd(d["answers"]) for d in subset]
        predictions = [clean_xsd(d["prediction"]) for d in subset]
        metrics = cal_metrics(golden_answers, predictions)
        values.append((metrics[2], metrics[8]))
    return (
        _fmt(values[0][0]),
        _fmt(values[1][0]),
        _fmt(values[2][0]),
        _fmt(values[2][1]),
        n,
    )


def _cwq_exact_match(
    golden_answers: list[list[str]], predictions: list[list[str]]
) -> float:
    if not predictions:
        return 0.0
    good = 0
    for golden, pred in zip(golden_answers, predictions):
        if set(golden) & set(pred):
            good += 1
    return round(100 * good / len(predictions), 2)


def _cwq_metrics(
    save_dir: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    num_workers: int | None = None,
) -> tuple[str, str, str, str, str, str, str]:
    data = _load_json_predictions(
        save_dir,
        dataset="cwq",
        only_sample=only_sample,
        dataset_base_dir=dataset_base_dir,
        num_workers=num_workers,
    )
    n = str(len(data))
    if not data:
        return "-", "-", "-", "-", "-", "-", n

    groups = [
        [d for d in data if d["compositionality_type"] == "conjunction"],
        [d for d in data if d["compositionality_type"] == "composition"],
        [d for d in data if d["compositionality_type"] == "comparative"],
        [d for d in data if d["compositionality_type"] == "superlative"],
        deepcopy(data),
    ]
    f1_values = []
    em_value = None
    for idx, subset in enumerate(groups):
        if not subset:
            f1_values.append(None)
            continue
        golden_answers = [clean_xsd(d["answers"]) for d in subset]
        predictions = [clean_xsd(d["prediction"]) for d in subset]
        metrics = cal_metrics(golden_answers, predictions)
        f1_values.append(metrics[2])
        if idx == len(groups) - 1:
            em_value = _cwq_exact_match(golden_answers, predictions)
    return (
        _fmt(f1_values[0]),
        _fmt(f1_values[1]),
        _fmt(f1_values[2]),
        _fmt(f1_values[3]),
        _fmt(f1_values[4]),
        _fmt(em_value),
        n,
    )


def _kqapro_acc(subset: list[dict]) -> float | None:
    if not subset:
        return None
    good = 0
    for item in subset:
        answers = item.get("answers", item.get("answer"))
        if isinstance(answers, bool):
            answers = [answers]
        golden = clean_xsd(answers)
        pred = clean_xsd(item["prediction"])
        if whether_equal(golden, pred):
            good += 1
    return round(100 * good / len(subset), 2)


def _kqapro_metrics(
    save_dir: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    num_workers: int | None = None,
) -> tuple[str, ...]:
    data = _load_json_predictions(
        save_dir,
        dataset="kqapro",
        only_sample=only_sample,
        dataset_base_dir=dataset_base_dir,
        num_workers=num_workers,
    )
    for item in data:
        _normalize_golden_answers(item)
        if "qtype" not in item and "type" in item:
            item["qtype"] = item["type"]
    n = str(len(data))
    if not data:
        return tuple(["-"] * len(KQAPRO_QTYPES) + ["-"]) + (n,)

    per_type = []
    for qtype in KQAPRO_QTYPES:
        subset = [d for d in data if d.get("qtype") == qtype]
        acc = _kqapro_acc(subset)
        per_type.append("-" if acc is None else f"{acc:.2f}")
    overall = _kqapro_acc(data)
    return tuple(per_type + ([f"{overall:.2f}"] if overall is not None else ["-"])) + (
        n,
    )


def _grailqa_metrics(
    save_dir: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    num_workers: int | None = None,
) -> tuple[str, str, str, str, str]:
    data = _load_json_predictions(
        save_dir,
        dataset="grailqa",
        only_sample=only_sample,
        dataset_base_dir=dataset_base_dir,
        num_workers=num_workers,
    )
    n = str(len(data))
    if not data:
        return "-", "-", "-", "-", n

    groups = [
        [d for d in data if d.get("level") == "i.i.d."],
        [d for d in data if d.get("level") == "compositional"],
        [d for d in data if d.get("level") == "zero-shot"],
        deepcopy(data),
    ]
    f1_values = []
    for subset in groups:
        if not subset:
            f1_values.append(None)
            continue
        ans_key = "answers" if "answers" in subset[0] else "answer"
        golden_answers = [clean_xsd(d[ans_key]) for d in subset]
        predictions = [clean_xsd(d["prediction"]) for d in subset]
        metrics = cal_metrics(golden_answers, predictions)
        f1_values.append(metrics[2])
    return (
        _fmt(f1_values[0]),
        _fmt(f1_values[1]),
        _fmt(f1_values[2]),
        _fmt(f1_values[3]),
        n,
    )


def _linear_sft_rows(
    root: str,
    actor_name: str,
    dataset: str,
    show_seeds: bool = False,
) -> list[tuple[str, str]]:
    if show_seeds:
        return [
            (
                f"SFT (seed{seed})",
                f"{root}/linear/{actor_name}-seed{seed}/{dataset}",
            )
            for seed in DEFAULT_SEEDS
        ]
    return [("SFT", f"{root}/linear/{actor_name}-seed1/{dataset}")]


def _build_table_body(
    rows: list[tuple[str, str | None]],
    metrics_fn,
    empty_width: int,
    num_workers: int | None = None,
) -> list[list[str]]:
    """Compute per-method row metrics in parallel; section headers (path=None) keep empty columns."""
    workers = (
        _default_num_workers() if num_workers is None else max(1, int(num_workers))
    )
    indexed = list(enumerate(rows))
    body: list[list[str] | None] = [None] * len(rows)

    def _one(item: tuple[int, tuple[str, str | None]]) -> tuple[int, list[str]]:
        idx, (label, path) = item
        if path is None:
            return idx, [label] + [""] * empty_width
        metrics = metrics_fn(path)
        return idx, [label, *metrics]

    if workers == 1 or len(rows) <= 1:
        for item in indexed:
            idx, row = _one(item)
            body[idx] = row
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, row in pool.map(_one, indexed):
                body[idx] = row
    return [row for row in body if row is not None]


def _print_webqsp(
    root: str,
    actor_name: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    show_seeds: bool = False,
    num_workers: int | None = None,
):
    rows = [
        ("Baseline", None),
        ("DecAF", f"{root}/baselines/DecAF/webqsp"),
        ("Linear Decision", None),
        ("GPT-4.1", f"{root}/linear/gpt-4.1-2025-04-14-seed42/webqsp"),
        *_linear_sft_rows(root, actor_name, "webqsp", show_seeds=show_seeds),
        (
            "--major vote",
            f"{root}/linear/{actor_name}-major-vote/webqsp",
        ),
        ("Tree Search", None),
        ("BFS", f"{root}/tree/bfs-{actor_name}-v1/webqsp"),
        ("DFS", f"{root}/tree/dfs-{actor_name}-v1/webqsp"),
        ("RandomWalk", f"{root}/tree/randomwalk-{actor_name}-v1/webqsp"),
        (
            "IG-Beam Search",
            f"{root}/tree-with-reward/ig-beam-Actor-{actor_name}-v1/webqsp",
        ),
        (
            "Classic MCTS",
            f"{root}/tree-with-reward/classic-mcts-Actor-{actor_name}-v1/webqsp",
        ),
        (
            "Fast MCTS",
            f"{root}/tree-with-reward/fast-mcts-Actor-{actor_name}-v1/webqsp",
        ),
        (
            "Fast MCTS-Const",
            f"{root}/tree/fast-mcts-constant-reward-{actor_name}-v1/webqsp",
        ),
    ]

    def metrics_fn(path: str):
        # Method rows run in parallel; read each directory sequentially to avoid nested thread pools saturating the machine.
        return _webqsp_metrics(
            path,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            num_workers=1,
        )

    headers = ["Method", "1-hop", "2-hop", "Overall", "RHits@1", "n"]
    body = _build_table_body(rows, metrics_fn, empty_width=5, num_workers=num_workers)

    print("- WebQSP:")
    print(_format_aligned_table(headers, body))
    print()


def _print_grailqa(
    root: str,
    actor_name: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    show_seeds: bool = False,
    num_workers: int | None = None,
):
    rows = [
        ("Baseline", None),
        ("DecAF", f"{root}/baselines/DecAF/grailqa"),
        ("Linear Decision", None),
        ("GPT-4.1", f"{root}/linear/gpt-4.1-2025-04-14-seed42/grailqa"),
        *_linear_sft_rows(root, actor_name, "grailqa", show_seeds=show_seeds),
        (
            "--major vote",
            f"{root}/linear/{actor_name}-major-vote/grailqa",
        ),
        ("Tree Search", None),
        ("BFS", f"{root}/tree/bfs-{actor_name}-v1/grailqa"),
        ("DFS", f"{root}/tree/dfs-{actor_name}-v1/grailqa"),
        ("RandomWalk", f"{root}/tree/randomwalk-{actor_name}-v1/grailqa"),
        (
            "IG-Beam Search",
            f"{root}/tree-with-reward/ig-beam-Actor-{actor_name}-v1/grailqa",
        ),
        (
            "Classic MCTS",
            f"{root}/tree-with-reward/classic-mcts-Actor-{actor_name}-v1/grailqa",
        ),
        (
            "Fast MCTS",
            f"{root}/tree-with-reward/fast-mcts-Actor-{actor_name}-v1/grailqa",
        ),
        (
            "Fast MCTS-Const",
            f"{root}/tree/fast-mcts-constant-reward-{actor_name}-v1/grailqa",
        ),
    ]

    def metrics_fn(path: str):
        return _grailqa_metrics(
            path,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            num_workers=1,
        )

    headers = ["Method", "i.i.d.", "Comp.", "Zero", "Overall", "n"]
    body = _build_table_body(rows, metrics_fn, empty_width=5, num_workers=num_workers)

    print("- GrailQA:")
    print(_format_aligned_table(headers, body))
    print()


def _print_cwq(
    root: str,
    actor_name: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    show_seeds: bool = False,
    num_workers: int | None = None,
):
    rows = [
        ("Baseline", None),
        ("DecAF", f"{root}/baselines/DecAF/cwq"),
        ("Linear Decision", None),
        ("GPT-4.1", f"{root}/linear/gpt-4.1-2025-04-14-seed42/cwq"),
        *_linear_sft_rows(root, actor_name, "cwq", show_seeds=show_seeds),
        (
            "--major vote",
            f"{root}/linear/{actor_name}-major-vote/cwq",
        ),
        ("Tree Search", None),
        ("BFS", f"{root}/tree/bfs-{actor_name}-v1/cwq"),
        ("DFS", f"{root}/tree/dfs-{actor_name}-v1/cwq"),
        ("RandomWalk", f"{root}/tree/randomwalk-{actor_name}-v1/cwq"),
        (
            "IG-Beam Search",
            f"{root}/tree-with-reward/ig-beam-Actor-{actor_name}-v1/cwq",
        ),
        (
            "Classic MCTS",
            f"{root}/tree-with-reward/classic-mcts-Actor-{actor_name}-v1/cwq",
        ),
        (
            "Fast MCTS",
            f"{root}/tree-with-reward/fast-mcts-Actor-{actor_name}-v1/cwq",
        ),
        (
            "Fast MCTS-Const",
            f"{root}/tree/fast-mcts-constant-reward-{actor_name}-v1/cwq",
        ),
    ]

    def metrics_fn(path: str):
        return _cwq_metrics(
            path,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            num_workers=1,
        )

    headers = ["Method", "Conj", "Compo", "Compa", "Super", "Overall", "EM", "n"]
    body = _build_table_body(rows, metrics_fn, empty_width=7, num_workers=num_workers)

    print("- CWQ:")
    print(_format_aligned_table(headers, body))
    print()


def _print_kqapro(
    root: str,
    actor_name: str,
    dataset_base_dir: str | None = None,
    only_sample: bool = False,
    show_seeds: bool = False,
    num_workers: int | None = None,
):
    rows = [
        ("Baseline", None),
        ("BART-SPARQL", f"{root}/baselines/BART-SPARQL/kqapro"),
        ("Linear Decision", None),
        ("GPT-4.1", f"{root}/linear/gpt-4.1-2025-04-14-seed42/kqapro"),
        *_linear_sft_rows(root, actor_name, "kqapro", show_seeds=show_seeds),
        (
            "--major vote",
            f"{root}/linear/{actor_name}-major-vote/kqapro",
        ),
        ("Tree Search", None),
        ("BFS", f"{root}/tree/bfs-{actor_name}-v1/kqapro"),
        ("DFS", f"{root}/tree/dfs-{actor_name}-v1/kqapro"),
        ("RandomWalk", f"{root}/tree/randomwalk-{actor_name}-v1/kqapro"),
        (
            "IG-Beam Search",
            f"{root}/tree-with-reward/ig-beam-Actor-{actor_name}-v1/kqapro",
        ),
        (
            "Classic MCTS",
            f"{root}/tree-with-reward/classic-mcts-Actor-{actor_name}-v1/kqapro",
        ),
        (
            "Fast MCTS",
            f"{root}/tree-with-reward/fast-mcts-Actor-{actor_name}-v1/kqapro",
        ),
        (
            "Fast MCTS-Const",
            f"{root}/tree/fast-mcts-constant-reward-{actor_name}-v1/kqapro",
        ),
    ]

    def metrics_fn(path: str):
        return _kqapro_metrics(
            path,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            num_workers=1,
        )

    headers = [
        "Method",
        *[KQAPRO_QTYPE_SHORT[qtype] for qtype in KQAPRO_QTYPES],
        "Overall",
        "n",
    ]
    body = _build_table_body(
        rows, metrics_fn, empty_width=len(headers) - 1, num_workers=num_workers
    )

    print("- KQA Pro:")
    print(_format_aligned_table(headers, body))
    print()


def run(
    root: str = "save-fast-mcts-kbqa",
    dataset: str = "all",
    actor_name: str = ACTOR_NAME,
    dataset_base_dir: str | None = DEFAULT_DATASET_BASE_DIR,
    only_sample: bool = False,
    show_seeds: bool = False,
    num_workers: int | None = None,
):
    dataset = dataset.lower()
    if num_workers is None:
        num_workers = _default_num_workers()
    if only_sample and dataset not in ("all", "webqsp", "cwq", "grailqa", "kqapro"):
        raise ValueError(
            f"only_sample only supports webqsp/cwq/grailqa/kqapro, got: {dataset}"
        )
    if dataset in ("all", "webqsp"):
        _print_webqsp(
            root,
            actor_name,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            show_seeds=show_seeds,
            num_workers=num_workers,
        )
    if dataset in ("all", "cwq"):
        _print_cwq(
            root,
            actor_name,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            show_seeds=show_seeds,
            num_workers=num_workers,
        )
    if dataset in ("all", "grailqa"):
        _print_grailqa(
            root,
            actor_name,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            show_seeds=show_seeds,
            num_workers=num_workers,
        )
    if dataset in ("all", "kqapro"):
        _print_kqapro(
            root,
            actor_name,
            dataset_base_dir=dataset_base_dir,
            only_sample=only_sample,
            show_seeds=show_seeds,
            num_workers=num_workers,
        )
    if dataset not in ("all", "webqsp", "cwq", "grailqa", "kqapro"):
        raise ValueError(f"unsupported dataset: {dataset}")


if __name__ == "__main__":
    fire.Fire(run)
