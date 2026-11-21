"""Classic MCTS for KBQA with terminal rollouts."""

import random
from functools import partial
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
from kb.actions import ToolInfraError
from search.base import ACTOR_MODEL, REWARD_MODEL, KBQATreeSearch, Node


class ClassicMCTSKBQA(KBQATreeSearch):
    """Estimate child values by rolling out to a terminal state."""

    def __init__(
        self,
        reward_method: str = "conditional_probability",
        *args,
        **kwargs,
    ):
        """Configure the reward used to evaluate terminal states."""
        super().__init__(*args, **kwargs)

        if reward_method == "conditional_probability":
            from reward.conditional_prob_sglang import compute_information_gain

            reward_function = partial(
                compute_information_gain, model_name=self.reward_model
            )
        elif reward_method == "logic_form":
            from reward.logic_form import compute_logic_form_reward as reward_function
        else:
            raise ValueError(f"reward_method: {reward_method} not supported.")

        self.reward_method = reward_method
        self.reward_function = reward_function

    def expand(self, node: Node, simulation_mode: bool = False) -> list[Node]:
        """Expand a tree leaf or sample one detached successor for a rollout."""
        messages, candidates = self._sample_candidates(
            node, temperature=0.7, max_tokens=512, rollout=simulation_mode
        )
        depth = sum(message["role"] == "assistant" for message in messages)
        history = messages.copy()
        children = []
        for content, observation in candidates:
            observation, is_terminal = self._resolve_done(
                content, observation, history, depth
            )
            child = self._create_child(
                node, content, observation, is_terminal=is_terminal
            )
            children.append(child)
            if simulation_mode:
                break

            node.children.append(child)
            if is_terminal:
                break

            # Classic's Done check also sees the preceding candidates from this expansion.
            history.extend(
                (
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Observation: {observation}"},
                )
            )
        return children

    def simulate(self, node: Node) -> float:
        """Follow a short rollout and score its terminal state when reached."""
        for _ in range(4):
            if node.is_terminal or len(node.path) >= self.max_depth:
                messages = self.path_to_complete_messages(node.path)
                reward = self.reward_function(messages=messages, q=self.question)
                self._reward_num += 1
                return reward
            successors = self.expand(node, simulation_mode=True)
            if not successors:
                return 0
            node = successors[0]
        return 0

    def search(self, num_simulations: int = 30) -> None:
        """Expand a UCT-selected leaf and backpropagate rollout or terminal rewards."""
        for i_simulation in range(num_simulations):
            print(
                f"ID: {self.qid}. Simulation {i_simulation} / {num_simulations}. Call num: {self._call_num}"
            )
            if self.terminal_count >= self.terminal_count_threshold:
                break
            leaf = self.select()
            if leaf.is_exhausted:
                break

            for child in self.expand(leaf):
                if child.is_terminal:
                    messages = self.path_to_complete_messages(leaf.path + [child])
                    reward = self.reward_function(messages=messages, q=self.question)
                    self._reward_num += 1
                    child.reward = reward
                elif child.reward == 0:
                    child.reward = self.simulate(child)
                self.backpropagate_with_reward(child.path, child.reward)

                print(child)
                print()

        self.end_time = time()


@catch_keyboard_interrupt
def run(
    dataset: str,
    note: str = "v1",
    actor_model: str = ACTOR_MODEL,
    actor_n: int = 3,
    num_simulations: int = 30,
    reduce_method: str = "vote",
    reward_method: str = "conditional_probability",
    max_workers: int = 8,
    case_timeout_seconds: float | None = DEFAULT_CASE_TIMEOUT_SECONDS,
    seed: int = 1,
    run_sample_num: int | None = None,
    debug: bool = False,
    full: bool = False,
    use_sample: bool = False,
    dataset_base_dir: str | None = None,
    save_root: str = "save-mcts-conventional",
    save_dir: str | None = None,
    result_subdir: str = "vote-json",
    pdf_subdir: str = "vote-pdf",
    reward_model: str = REWARD_MODEL,
):
    """Run the rollout baseline and resume from existing predictions."""
    assert reward_method in [
        "conditional_probability",
        "logic_form",
    ], "reward_method must be one of conditional_probability, logic_form"
    _actor = actor_model.split("/")[-1]
    _setting = (
        f"Actor-{_actor}-Reward-{reward_method}"
        if not debug
        else f"Debug-Actor-{_actor}-Reward-{reward_method}"
    )

    save_dir = save_dir or f"{save_root}/{_setting}-{note}/{dataset}"
    result_dir = f"{save_dir}/{result_subdir}" if result_subdir else save_dir

    # Resume from questions with saved predictions.
    skip_ids = {
        path.split("/")[-1].split(".json")[0] for path in glob(f"{result_dir}/*.json")
    }
    logger.info(f"skip ids: {len(skip_ids)}")

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
        """Retry a question with a new seed until it has a valid terminal answer."""
        answer = d["answers"] if "answers" in d else d["answer"]
        if isinstance(answer, bool):
            answer = [answer]
        for i in range(5):
            qtype_key = get_qtype_key(dataset)
            mcts = ClassicMCTSKBQA(
                qid=d["id"],
                question=d["question"],
                golden_sparql=d["sparql"],
                golden_answer=answer,
                dataset=dataset,
                qtype=d[qtype_key],
                terminal_count_threshold=5,
                actor_model=actor_model,
                actor_n=actor_n,
                reward_method=reward_method,
                reward_model=reward_model,
                debug=debug,
                start_seed=seed + i,
            )
            try:
                mcts.search(num_simulations=num_simulations)
                if mcts.get_valid_done_paths():
                    mcts.cal_f1()
                    d = mcts.get_answer_by_obs_vote()
                    save_to_json(
                        d,
                        f"{result_dir}/{mcts.qid}.json",
                        _print=False,
                    )
                    try:
                        mcts.save(save_dir=f"{save_dir}/pkl")
                    except Exception as e:
                        logger.error(f"PKL save failed in id: {mcts.qid}, {e}")
                        print_exc()
                    break
                else:
                    logger.warning(f"No valid done paths in id: {d['id']}")
            except ToolInfraError as e:
                logger.error(f"Tool infra error in id: {d['id']}, {e}")
                d["tool_error"] = str(e)
                save_to_json(
                    d,
                    f"{result_dir}/{d['id']}.json",
                    _print=False,
                )
                break
            except Exception as e:
                logger.error(f"Error in id: {d['id']}, {e}")
                print_exc()

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
            desc="Running Classic MCTS",
        )
        logger.info(
            f"All done. total={stats['total']}, success={stats['success']}, "
            f"timeouts={stats['timeouts']}, failed={stats['errors']}"
        )


if __name__ == "__main__":
    if is_vscode_debug_mode():
        logger.debug("Debug mode.")
        run(
            "kqapro",
            note="debug",
            debug=True,
        )
    else:
        fire.Fire(run)
