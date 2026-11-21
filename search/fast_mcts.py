"""Fast MCTS for KBQA with information-gain rewards."""

import random
from glob import glob
from time import time
from traceback import print_exc

import fire
from loguru import logger

from common.common_utils import (
    DEFAULT_CASE_TIMEOUT_SECONDS,
    catch_keyboard_interrupt,
    is_vscode_debug_mode,
    run_thread_pool_cases,
    save_to_json,
)
from data_loader.load import get_qtype_key, load_test_data
from search.base import ACTOR_MODEL, REWARD_MODEL, KBQATreeSearch


class FastMCTSKBQA(KBQATreeSearch):
    """Score each child immediately with information gain and skip rollouts."""

    def __init__(self, *args, **kwargs):
        """Set up the information-gain scorer or its constant-reward ablation."""
        super().__init__(*args, **kwargs)
        self._configure_ig_reward()

    def search(self, num_simulations: int = 50) -> None:
        """Select a leaf with UCT and backpropagate the scores of its children."""
        for _ in range(num_simulations):
            if self.terminal_count >= self.terminal_count_threshold:
                break
            self.expand_and_backpropagate(self.select())
        self.end_time = time()


@catch_keyboard_interrupt
def run(
    dataset: str,
    note: str = "v1",
    actor_model: str = ACTOR_MODEL,
    reward_model: str = REWARD_MODEL,
    actor_n: int = 5,
    num_simulations: int = 50,
    reduce_method: str = "vote",
    max_workers: int = 16,
    case_timeout_seconds: float | None = DEFAULT_CASE_TIMEOUT_SECONDS,
    seed: int = 1,
    debug: bool = False,
    run_sample_num: int | None = None,
    full: bool = False,
    use_sample: bool = False,
    dataset_base_dir: str | None = None,
    save_root: str = "save-mcts",
    save_dir: str | None = None,
    result_subdir: str = "vote-json",
    pdf_subdir: str = "vote-pdf",
    use_constant_reward: bool = False,
):
    """Run Fast MCTS for a dataset and resume from existing predictions."""
    _actor = actor_model.split("/")[-1]
    _setting = f"Actor-{_actor}" if not debug else f"Debug-Actor-{_actor}"

    save_dir = save_dir or f"{save_root}/{_setting}-{note}/{dataset}"
    logger.info(f"Save dir: {save_dir}")
    result_dir = f"{save_dir}/{result_subdir}" if result_subdir else save_dir

    # Resume from final predictions, even if a previous run saved only a checkpoint.
    skip_ids = {
        path.split("/")[-1].split(".json")[0] for path in glob(f"{result_dir}/*.json")
    }
    logger.info(f"skip ids by json: {len(skip_ids)}")

    test_data = load_test_data(
        dataset,
        full=full or "fullTest" in note,
        dataset_base_dir=dataset_base_dir,
        use_sample=use_sample,
    )

    if run_sample_num is not None:
        random.seed(123)
        random.shuffle(test_data)
        test_data = test_data[:run_sample_num]
        logger.info(f"Run shuffle and sample {run_sample_num} items.")

    test_data = [d for d in test_data if str(d["id"]) not in skip_ids]
    logger.warning(f"Remaining test data: {len(test_data)}")

    def process_data(d):
        """Run one search and write its voted answer or fallback."""
        qtype_key = get_qtype_key(dataset)
        answer = d["answers"] if "answers" in d else d["answer"]
        if isinstance(answer, bool):
            answer = [answer]
        assert isinstance(answer, list)

        # Run one full search per question so samples without a valid Done are not searched again.
        mcts = FastMCTSKBQA(
            qid=d["id"],
            question=d["question"],
            golden_sparql=d["sparql"],
            golden_answer=answer,
            dataset=dataset,
            qtype=d[qtype_key],
            terminal_count_threshold=5,
            actor_model=actor_model,
            actor_n=actor_n,
            reward_model=reward_model,
            use_constant_reward=use_constant_reward,
            debug=debug,
            start_seed=seed,
        )
        try:
            mcts.search(num_simulations=num_simulations)
            if mcts.get_valid_done_paths():
                mcts.cal_f1()
                result = mcts.get_answer_by_obs_vote()
            else:
                logger.warning(
                    f"No valid done paths in id: {d['id']}; saving a random fallback path."
                )
                result = mcts.get_random_fallback_answer()
            save_to_json(result, f"{result_dir}/{mcts.qid}.json", _print=False)
        except Exception as e:
            logger.error(f"Error in id: {d['id']}, {e}")
            print_exc()
            # Save a checkpoint result on tool-call failures so the next resume does not retry timed-out samples forever.
            mcts.end_time = time()
            fallback = mcts.get_random_fallback_answer()
            save_to_json(fallback, f"{result_dir}/{mcts.qid}.json", _print=False)

    if debug:
        for d in test_data:
            process_data(d)
            break
    else:
        logger.warning(f"max_workers: {max_workers}")
        stats = run_thread_pool_cases(
            test_data,
            process_data,
            max_workers=max_workers,
            case_timeout_seconds=case_timeout_seconds,
            desc="Running Fast MCTS",
        )
        logger.info(
            f"All done. total={stats['total']}, success={stats['success']}, "
            f"timeouts={stats['timeouts']}, failed={stats['errors']}"
        )


if __name__ == "__main__":
    if is_vscode_debug_mode():
        logger.debug("Debug mode.")
        actor_model = (
            "save-model-sft/small-GrailQA-v2-v1/Meta-Llama-3.1-8B-Instruct-epoch5"
        )
        reward_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        run(
            dataset="grailqa",
            note="v1",
            debug=True,
            actor_model=actor_model,
            reward_model=reward_model,
        )
    else:
        fire.Fire(run)
