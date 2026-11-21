from glob import glob
from time import time
from traceback import print_exc
from typing import List

import fire
from loguru import logger

from common.common_utils import (
    DEFAULT_CASE_TIMEOUT_SECONDS,
    catch_keyboard_interrupt,
    is_vscode_debug_mode,
    run_thread_pool_cases,
    save_to_json,
)
from common.constant import ACTOR_KEY
from data_loader.load import get_qtype_key, load_test_data
from kb.actions import ToolInfraError, parse_action, should_skip_observation
from search.base import ACTOR_MODEL, REWARD_MODEL, KBQATreeSearch, Node


class IGBeamSearch(KBQATreeSearch):
    """IG-Beam Search: replace MCTS UCT selection with layer-wise beam pruning."""

    def __init__(
        self,
        *args,
        beam_width: int = 5,
        actor_call_budget: int = 50,
        **kwargs,
    ):
        """Set up the beam limits and its information-gain scorer."""
        super().__init__(*args, **kwargs)
        self._configure_ig_reward()
        self.beam_width = beam_width
        self.actor_call_budget = actor_call_budget

    @staticmethod
    def _action_key(content: str) -> str | None:
        action = parse_action(content, execute=False)
        if should_skip_observation(action):
            return None
        return action

    def _path_action_keys(self, node: Node) -> set[str]:
        keys = set()
        for path_node in node.path:
            key = self._action_key(path_node.content)
            if key and key != "Done":
                keys.add(key)
        return keys

    def _is_valid_done(self, messages: list[dict]) -> bool:
        """Return whether Done corresponds to an executable, non-empty final ExecuteSPARQL result."""
        if len(messages) < 2:
            return False

        last_action = (
            messages[-2]["content"].split("Action: ")[1].strip()
            if "Action: " in messages[-2]["content"]
            else "_None"
        )
        last_obs = messages[-1]["content"].split("Observation: ")[-1].strip()

        return (
            "ExecuteSPARQL" in last_action
            and last_obs != "[]"
            and "Error: " not in last_obs
            and "pred:fact_" not in last_obs
            and "?e <" not in last_obs
        )

    def _score_child(self, parent: Node, content: str, obs: str) -> Node | None:
        """Score child state after a candidate action; valid Done collects parent as terminal answer."""
        messages = self.path_to_complete_messages(parent.path)
        is_done = "Action: Done" in content

        if is_done:
            if not self._is_valid_done(messages):
                return None

            # For vote-json eval compatibility, record terminal answers at the ExecuteSPARQL state before Done.
            parent.is_terminal = True
            self.terminal_count += 1
            self.valid_done_nodes.append(parent)
            logger.warning(f"Node {parent.name} is terminal.")
            return None

        child_messages = messages.copy()
        child_messages.append({"role": "assistant", "content": content})
        child_messages.append({"role": "user", "content": f"Observation: {obs}"})

        reward = self.reward_function(
            messages=child_messages,
            q=self.question,
            model_name=self.reward_model,
        )
        self._reward_num += 1

        child = Node(
            name=f"Node-{self.node_index}",
            content=content,
            observation=obs,
            parent=parent,
            reward=reward,
        )
        self.node_index += 1
        parent.children.append(child)
        return child

    def _expand_one_state(self, node: Node) -> List[Node]:
        """Sample n actor candidates for one beam state and return child states to continue search."""
        if (
            node.is_terminal
            or node.is_exhausted
            or self._call_num >= self.actor_call_budget
        ):
            return []
        if len(node.path) >= self.max_depth:
            node.is_exhausted = True
            return []

        messages = self.path_to_complete_messages(node.path)
        response = self.actor_client.chat.completions.create(
            model=self.actor_model,
            messages=messages,
            temperature=1,
            top_p=0.95,
            n=self.actor_n,
            stop=["\nObservation", "\nThought"],
            max_tokens=1024,
        )
        self._call_num += 1

        choices = [
            choice.message.content
            for choice in response.choices
            if choice.message and choice.message.content
        ]
        choices = [c.split("Observation")[0].strip() for c in choices if c]

        children = []
        visited_obs = set()
        visited_actions = self._path_action_keys(node)
        depth = sum(m["role"] == "assistant" for m in messages)
        for choice in choices:
            action_key = self._action_key(choice)
            if action_key is None or action_key in visited_actions:
                continue
            visited_actions.add(action_key)

            obs = parse_action(f"Action: {action_key}", db=self.db, execute=True)
            if obs is None or should_skip_observation(obs) or obs in visited_obs:
                continue
            visited_obs.add(obs)

            child = self._score_child(node, choice, obs)
            if child is not None:
                children.append(child)

            # Stop wasting reward calls on this sample once enough terminal answers are found.
            if self.terminal_count >= self.terminal_count_threshold:
                break

        logger.info(f"Expanded {len(children)} children at depth {depth}.")
        if not children and not node.is_terminal:
            node.expand_fail_count += 1
            if node.expand_fail_count >= self.expand_failure_threshold:
                node.is_exhausted = True
        return children

    def _select_next_beam(self, candidates: List[Node]) -> List[Node]:
        """Prefer parent diversity, then fill remaining slots by reward."""
        sorted_candidates = sorted(candidates, key=lambda n: n.reward, reverse=True)
        if len(sorted_candidates) <= self.beam_width:
            return sorted_candidates

        selected = []
        selected_ids = set()
        seen_parents = set()

        for child in sorted_candidates:
            parent_name = child.parent.name if child.parent else ""
            if parent_name in seen_parents:
                continue
            selected.append(child)
            selected_ids.add(id(child))
            seen_parents.add(parent_name)
            if len(selected) >= self.beam_width:
                return selected

        for child in sorted_candidates:
            if id(child) in selected_ids:
                continue
            selected.append(child)
            if len(selected) >= self.beam_width:
                break

        return selected

    def search(self, max_depth: int | None = None):
        """Expand the current beam layer by layer and keep top-B child states by IG reward."""
        if max_depth is None:
            max_depth = self.max_depth

        beam = [self.root]
        for depth in range(max_depth):
            if (
                not beam
                or self.terminal_count >= self.terminal_count_threshold
                or self._call_num >= self.actor_call_budget
            ):
                break

            next_candidates = []
            for node in beam:
                next_candidates.extend(self._expand_one_state(node))
                if (
                    self.terminal_count >= self.terminal_count_threshold
                    or self._call_num >= self.actor_call_budget
                ):
                    break

            # Beam pruning: ensure parent-branch coverage first, then fill top-B with IG reward.
            beam = self._select_next_beam(next_candidates)
            logger.info(
                f"Beam depth {depth + 1}: keep {len(beam)} states, "
                f"actor calls={self._call_num}, terminals={self.terminal_count}."
            )

        self.end_time = time()


@catch_keyboard_interrupt
def run(
    dataset: str,
    note: str = "v1",
    actor_model: str = ACTOR_MODEL,
    reward_model: str = REWARD_MODEL,
    actor_n: int = 5,
    beam_width: int = 5,
    actor_call_budget: int = 50,
    terminal_count_threshold: int = 5,
    max_depth: int | None = None,
    reduce_method: str = "vote",
    max_workers: int = 16,
    case_timeout_seconds: float | None = DEFAULT_CASE_TIMEOUT_SECONDS,
    debug: bool = False,
    run_sample_num: int | None = None,
    full: bool = False,
    use_sample: bool = False,
    dataset_base_dir: str | None = None,
    save_root: str = "save-mcts",
    save_dir: str | None = None,
    result_subdir: str = "vote-json",
):
    _actor = actor_model.split("/")[-1]
    _setting = f"IG-Beam-Actor-{_actor}-Reward-conditional_probability"
    if debug:
        _setting = f"Debug-{_setting}"

    save_dir = save_dir or f"{save_root}/{_setting}-{note}/{dataset}"
    logger.info(f"Save dir: {save_dir}")
    result_dir = f"{save_dir}/{result_subdir}" if result_subdir else save_dir

    skip_ids = set()
    for p in glob(f"{result_dir}/*.json"):
        skip_ids.add(p.split("/")[-1].split(".json")[0])
    logger.info(f"Load {len(skip_ids)} existed ids.")

    test_data = load_test_data(
        dataset,
        full=full or "fullTest" in note,
        dataset_base_dir=dataset_base_dir,
        use_sample=use_sample,
    )
    if run_sample_num is not None:
        import random

        random.seed(123)
        random.shuffle(test_data)
        test_data = test_data[:run_sample_num]
        logger.info(f"Run shuffle and sample {run_sample_num} items.")

    test_data = [d for d in test_data if str(d["id"]) not in skip_ids]
    logger.warning(f"Remaining test data: {len(test_data)}")

    def process_data(d):
        qtype_key = get_qtype_key(dataset)
        answer = d["answers"] if "answers" in d else d["answer"]
        if isinstance(answer, bool):
            answer = [answer]
        assert isinstance(answer, list)

        for i in range(5):
            beam_search = IGBeamSearch(
                qid=d["id"],
                question=d["question"],
                golden_sparql=d["sparql"],
                golden_answer=answer,
                dataset=dataset,
                qtype=d[qtype_key],
                terminal_count_threshold=terminal_count_threshold,
                actor_model=actor_model,
                actor_n=actor_n,
                reward_model=reward_model,
                debug=debug,
                start_seed=None,
                beam_width=beam_width,
                actor_call_budget=actor_call_budget,
            )
            try:
                beam_search.search(max_depth=max_depth)
                if beam_search.get_valid_done_paths():
                    beam_search.cal_f1()
                    result = beam_search.get_answer_by_obs_vote()
                    save_to_json(
                        result,
                        f"{result_dir}/{beam_search.qid}.json",
                        _print=False,
                    )
                    try:
                        beam_search.save(save_dir=f"{save_dir}/pkl")
                    except Exception as e:
                        logger.error(f"PKL save failed in id: {beam_search.qid}, {e}")
                        print_exc()
                    break
                logger.warning(f"No valid done paths in id: {d['id']} at try {i}.")
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
            desc="Running IG-Beam Search",
        )
        logger.info(
            f"All done. total={stats['total']}, success={stats['success']}, "
            f"timeouts={stats['timeouts']}, failed={stats['errors']}"
        )


if __name__ == "__main__":
    if is_vscode_debug_mode():
        logger.debug("Debug mode.")
        run(dataset="cwq", note="debug", debug=True, run_sample_num=1)
    else:
        fire.Fire(run)
