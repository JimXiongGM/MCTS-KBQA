from __future__ import annotations

import os
from functools import lru_cache
from glob import glob

from common.common_utils import read_json

DATASET_BASE_CANDIDATES = ("data",)

V3_DATASET_DIRS = {
    "webqsp": "WebQSP",
    "cwq": "CWQ",
    "kqapro": "KQAPro",
    "grailqa": "GrailQA",
}

V3_DATASET_FILES = {
    "webqsp": "test.json",
    "cwq": "test.json",
    "kqapro": "test.json",
    "grailqa": "dev.json",
}

SAMPLE_INFO_FILENAME = "sample_info.json"
SAMPLE_DATASETS = frozenset({"webqsp", "cwq", "grailqa"})
KQAPRO_TEST_SIZE = 900


def resolve_dataset_base_dir(dataset_base_dir: str | None = None) -> str:
    if dataset_base_dir:
        if os.path.isdir(dataset_base_dir):
            return dataset_base_dir.rstrip("/")
        raise FileNotFoundError(f"dataset base dir not found: {dataset_base_dir}")

    env_dir = os.environ.get("KBQA_DATASET_BASE_DIR")
    if env_dir:
        if os.path.isdir(env_dir):
            return env_dir.rstrip("/")
        raise FileNotFoundError(f"KBQA_DATASET_BASE_DIR does not exist: {env_dir}")

    for candidate in DATASET_BASE_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        "No dataset base dir found. Tried: " + ", ".join(DATASET_BASE_CANDIDATES)
    )


def _dataset_split_dir(dataset: str, full: bool) -> str:
    if dataset == "webqsp":
        return "test" if full else "test-300"
    if dataset == "cwq":
        return "test" if full else "test-600"
    if dataset == "grailqa":
        return "test" if full else "dev-600"
    raise ValueError(f"dataset: {dataset} not supported.")


def get_dataset_split_glob(
    dataset: str,
    full: bool = False,
    dataset_base_dir: str | None = None,
) -> str:
    """Locate the released evaluation files and fall back to per-dataset split layouts."""
    base_dir = resolve_dataset_base_dir(dataset_base_dir)
    v3_file = (
        f"{base_dir}/{V3_DATASET_DIRS[dataset]}/{V3_DATASET_FILES[dataset]}"
        if dataset in V3_DATASET_DIRS
        else None
    )
    if v3_file and (dataset == "kqapro" or os.path.isfile(v3_file)):
        return v3_file
    return f"{base_dir}/{dataset}/{_dataset_split_dir(dataset, full)}/*.json"


def load_kqapro_test_data(dataset_base_dir: str | None = None) -> list[dict]:
    """Load and validate the fixed 900-question KQA Pro evaluation set."""
    path = get_dataset_split_glob("kqapro", dataset_base_dir=dataset_base_dir)
    items = read_json(path)
    ids = {str(item["id"]) for item in items}
    if len(items) != KQAPRO_TEST_SIZE or len(ids) != KQAPRO_TEST_SIZE:
        raise ValueError(
            f"Expected {KQAPRO_TEST_SIZE} unique KQA Pro questions in {path}"
        )
    return items


@lru_cache(maxsize=8)
def load_kqapro_test_ids(dataset_base_dir: str | None = None) -> frozenset[str]:
    """Return the fixed question IDs used by evaluation and voting."""
    return frozenset(
        str(item["id"]) for item in load_kqapro_test_data(dataset_base_dir)
    )


def list_dataset_split_files(
    dataset: str,
    full: bool = False,
    dataset_base_dir: str | None = None,
) -> list[str]:
    return glob(get_dataset_split_glob(dataset, full, dataset_base_dir))


def get_sample_info_path(dataset: str, dataset_base_dir: str | None = None) -> str:
    if dataset not in SAMPLE_DATASETS:
        raise ValueError(
            f"use_sample only supports {sorted(SAMPLE_DATASETS)}, got: {dataset}"
        )
    base_dir = resolve_dataset_base_dir(dataset_base_dir)
    path = f"{base_dir}/{V3_DATASET_DIRS[dataset]}/{SAMPLE_INFO_FILENAME}"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"sample_info not found: {path}")
    return path


@lru_cache(maxsize=16)
def load_sample_ids(
    dataset: str, dataset_base_dir: str | None = None
) -> frozenset[str]:
    path = get_sample_info_path(dataset, dataset_base_dir)
    info = read_json(path)
    sample_ids = info.get("sample_ids")
    if not sample_ids:
        raise ValueError(f"sample_ids is empty in {path}")
    return frozenset(str(i) for i in sample_ids)


def filter_by_sample_ids(
    items: list,
    dataset: str,
    use_sample: bool = False,
    dataset_base_dir: str | None = None,
    id_key: str = "id",
) -> list:
    if not use_sample or dataset not in SAMPLE_DATASETS:
        return items
    sample_ids = load_sample_ids(dataset, dataset_base_dir)
    filtered = [item for item in items if str(item.get(id_key, "")) in sample_ids]
    if not filtered:
        raise ValueError(
            f"no items left after use_sample filter for {dataset}, "
            f"sample_ids={len(sample_ids)}"
        )
    return filtered
