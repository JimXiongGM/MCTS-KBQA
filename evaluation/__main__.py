from pathlib import Path

import fire


def run(
    save_dir,
    full=False,
    force_full=False,
    use_sample=False,
    dataset_base_dir=None,
    dataset=None,
):
    """Run the dataset evaluator, inferring the dataset from the predictions path when omitted."""
    if dataset is None:
        for part in reversed(Path(save_dir).parts):
            dataset = next(
                (
                    name
                    for name in ("webqsp", "cwq", "kqapro", "grailqa")
                    if name in part.lower()
                ),
                None,
            )
            if dataset:
                break
    if dataset == "webqsp":
        from evaluation.runtime import evaluation_webqsp as eval_func
    elif dataset == "cwq":
        from evaluation.runtime import evaluation_cwq as eval_func
    elif dataset == "kqapro":
        from evaluation.runtime import evaluation_kqapro as eval_func
    elif dataset == "grailqa":
        from evaluation.runtime import evaluation_grailqa as eval_func
    else:
        raise ValueError(f"Unknown dataset for {save_dir}. Specify --dataset.")

    if dataset == "kqapro":
        eval_func(save_dir, dataset_base_dir=dataset_base_dir)
    else:
        eval_func(
            save_dir,
            full=full,
            force_full=force_full,
            use_sample=use_sample,
            dataset_base_dir=dataset_base_dir,
        )


if __name__ == "__main__":
    """
    python -m evaluation --save_dir final-mcts/Rsample-v1/webqsp-maxR
    """
    from common.common_utils import is_vscode_debug_mode

    if is_vscode_debug_mode():
        # run("save-interactive/Meta-Llama-3.1-8B-Instruct-epoch5-seed1/kqapro")
        # run("save-mcgs/Actor-Meta-Llama-3.1-8B-Instruct-epoch5-v1/kqapro/vote-json")
        run("save-interactive/Meta-Llama-3.1-8B-Instruct-epoch5-seed42/grailqa")
    else:
        fire.Fire(run)
