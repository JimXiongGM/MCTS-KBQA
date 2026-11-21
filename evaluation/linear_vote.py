import json
import os
from collections import Counter
from glob import glob

from common.path_utils import load_kqapro_test_ids, load_sample_ids


def main(
    path,
    dataset,
    output_path=None,
    seed_values="1,10,100,1000,10000",
    use_sample=False,
    dataset_base_dir=None,
):
    """Vote across random seeds for the same evaluation question."""
    seeds = [s.strip() for s in seed_values.split(",") if s.strip()]
    if not seeds:
        raise ValueError("seed_values is empty")

    if dataset == "grailqa" or "grailqa" in path:

        def extract_answer(d):
            return d["prediction"]

    elif dataset == "kqapro" or "kqa" in path:
        from evaluation.kqapro import extract_answer
    else:
        from evaluation.runtime import (
            extract_dialog_last_valid_observation as extract_answer,
        )

    seed1 = seeds[0]
    if not path.endswith(f"seed{seed1}"):
        raise ValueError(f"path must end with seed{seed1}, got: {path}")

    output_dir = output_path or f"{path[:-len(f'seed{seed1}')]}vote5/{dataset}"
    os.makedirs(output_dir, exist_ok=True)

    sample_ids = None
    if dataset == "kqapro":
        sample_ids = load_kqapro_test_ids(dataset_base_dir)
    elif use_sample and dataset in ("webqsp", "cwq"):
        sample_ids = load_sample_ids(dataset, dataset_base_dir)

    c = 0
    for p in glob(f"{path}/{dataset}/*.json"):
        qid = os.path.basename(p).split(".json")[0]
        if sample_ids is not None and qid not in sample_ids:
            continue
        all_paths = [p.replace(f"seed{seed1}", f"seed{seed}") for seed in seeds]
        d_s = [json.load(open(path_i)) for path_i in all_paths]
        answer_docs = [(str(extract_answer(d)), d) for d in d_s]
        ans_s = [answer for answer, _ in answer_docs if answer != "['_None']"]

        save_p = f"{output_dir}/{d_s[0]['id']}.json"
        if ans_s:
            prediction = Counter(ans_s).most_common()[0][0]
            for a, d in answer_docs:
                if a == prediction:
                    with open(save_p, "w") as f:
                        json.dump(d, f, ensure_ascii=False, indent=4)
                        c += 1
                        break
        else:
            with open(save_p, "w") as f:
                json.dump(d_s[0], f, ensure_ascii=False, indent=4)
                c += 1

    print(f"Save {c} files.")


if __name__ == "__main__":
    """
    python -m evaluation.linear_vote \
      --path save-fast-mcts-kbqa/linear/Meta-Llama-3.1-8B-Instruct-epoch5-seed1 \
      --dataset webqsp \
      --output_path save-fast-mcts-kbqa/linear/Meta-Llama-3.1-8B-Instruct-epoch5-major-vote/webqsp
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path argument")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset argument")
    parser.add_argument(
        "--output_path", type=str, default=None, help="Output directory"
    )
    parser.add_argument(
        "--seed_values",
        type=str,
        default="1,10,100,1000,10000",
        help="Comma-separated list of seeds",
    )
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Only process ids in sample_info.json (webqsp/cwq)",
    )
    parser.add_argument(
        "--dataset_base_dir",
        type=str,
        default=None,
        help="Dataset root directory; defaults to KBQA_DATASET_BASE_DIR or data/",
    )
    args = parser.parse_args()

    main(
        args.path,
        args.dataset,
        args.output_path,
        args.seed_values,
        use_sample=args.use_sample,
        dataset_base_dir=args.dataset_base_dir,
    )
