import random
from collections import deque
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
from search.base import ACTOR_MODEL, KBQATreeSearch, Node


class TreeTraversal(KBQATreeSearch):
    def __init__(self, *args, **kwargs):
        """Initialize an unscored traversal using constant child rewards."""
        super().__init__(*args, **kwargs)

        # Tree Traversal does not need reward function.
        self.reward_mode = "cond_prob"
        self.reward_function = self.constant_reward

    def select(self):
        raise NotImplementedError

    def simulate(self):
        raise NotImplementedError

    def backpropagate(self, *args, **kwargs):
        pass

    def search(self, max_run_num: int = 30):
        """Expand selected nodes up to the traversal or terminal limit."""
        for _ in range(max_run_num):
            if self.terminal_count >= self.terminal_count_threshold:
                break
            leaf = self.select()
            if leaf is not None:
                self.expand_and_backpropagate(leaf)
        self.end_time = time()


class DFS(TreeTraversal):
    def select(self, node: Node = None) -> Node:
        """
        DFS: Select the leftmost valid node
        A node is valid if:
        1. It is a leaf node
        2. It is not terminal
        Otherwise, backtrack until a valid node is found
        """
        if node is None:
            node = self.root
        if len(node.path) >= self.max_depth:
            return None

        # If the current node is a leaf node and is not terminal, return directly
        if node.is_leaf:
            if node.is_terminal:
                return None
            else:
                return node

        # If the current node has children, recursively check the leftmost child
        for child in node.children:
            r = self.select(child)
            if r is not None:
                return r

        return None


class BFS(TreeTraversal):
    def select(self) -> Node:
        """
        BFS: Use a queue to traverse level by level, selecting the first valid node
        A node is valid if:
        1. It is a leaf node
        2. It is not terminal
        Otherwise, continue to traverse until a valid node is found
        """
        # Use a queue for BFS
        queue = deque([self.root])

        while queue:
            curr = queue.popleft()

            # If the current node is a leaf node and is not terminal, return directly
            if curr.is_leaf:
                if curr.is_terminal:
                    continue
                else:
                    return curr

            # Add all children to the queue
            queue.extend(curr.children)

        return None


class RandomWalk(TreeTraversal):
    def select(self, node: Node = None) -> Node:
        if node is None:
            node = self.root
        if len(node.path) >= self.max_depth:
            return None

        # If the current node is a leaf node and is not terminal, return directly
        if node.is_leaf:
            if node.is_terminal:
                return None
            else:
                return node

        nodes = node.children
        random.shuffle(nodes)
        for node in nodes:
            r = self.select(node)
            if r is not None:
                return r
        return None


@catch_keyboard_interrupt
def run_tree_search(
    dataset: str,
    note: str = "v1",
    actor_model: str = ACTOR_MODEL,
    actor_n: int = 5,
    select_method: str = "dfs",
    max_run_num: int = 50,
    max_workers: int = 8,
    case_timeout_seconds: float | None = DEFAULT_CASE_TIMEOUT_SECONDS,
    debug: bool = False,
    full: bool = False,
    use_sample: bool = False,
    dataset_base_dir: str | None = None,
    save_root: str = "save-mcts",
    save_dir: str | None = None,
    result_subdir: str = "json",
):

    if select_method == "dfs":
        search_class = DFS
    elif select_method == "bfs":
        search_class = BFS
    elif select_method == "randomwalk":
        search_class = RandomWalk
    else:
        raise ValueError(f"Invalid method: {select_method}")

    logger.info(f"select_method: {select_method}")

    _actor = actor_model.split("/")[-1]
    save_dir = (
        save_dir or f"{save_root}/{select_method}-Actor-{_actor}-{note}/{dataset}"
    )
    logger.info(f"save_dir: {save_dir}")
    result_dir = f"{save_dir}/{result_subdir}" if result_subdir else save_dir

    # Resume based on final JSON outputs. A previous run may have written PKL but
    # been interrupted before JSON was saved.
    skip_ids = set()
    for p in glob(f"{result_dir}/*.json"):
        skip_ids.add(p.split("/")[-1].split(".json")[0])
    logger.info(f"skip ids by json: {len(skip_ids)}")

    test_data = load_test_data(
        dataset,
        full=full,
        dataset_base_dir=dataset_base_dir,
        use_sample=use_sample,
    )

    test_data = [d for d in test_data if str(d["id"]) not in skip_ids]
    logger.warning(f"Remaining test data: {len(test_data)}")

    # debug
    if debug:
        # test_data = [d for d in test_data if d["id"] == "WebQTest-516.P0"]
        test_data = test_data[:10]

    def process_data(d):
        for i in range(5):
            qtype_key = get_qtype_key(dataset)
            answer = d["answers"] if "answers" in d else d["answer"]
            if isinstance(answer, bool):
                answer = [answer]
            tree = search_class(
                qid=d["id"],
                question=d["question"],
                golden_sparql=d["sparql"],
                golden_answer=answer,
                dataset=dataset,
                qtype=d[qtype_key],
                terminal_count_threshold=5,
                actor_model=actor_model,
                actor_n=actor_n,
                debug=debug,
                start_seed=i,
            )
            try:
                tree.search(max_run_num=max_run_num)
                tree.cal_f1()
                if tree.get_valid_done_paths():
                    d = tree.get_answer_by_obs_vote()
                    save_to_json(d, f"{result_dir}/{tree.qid}.json", _print=False)
                    try:
                        tree.save(save_dir=f"{save_dir}/pkl")
                    except Exception as e:
                        logger.error(f"PKL save failed in id: {tree.qid}, {e}")
                        print_exc()
                    break
                else:
                    logger.warning(f"No valid done paths in id: {d['id']} at try {i}.")
            except ToolInfraError as e:
                logger.error(f"Tool infra error in id: {d['id']}, {e}")
                d["tool_error"] = str(e)
                save_to_json(d, f"{result_dir}/{d['id']}.json", _print=False)
                break
            except Exception as e:
                logger.error(f"Error in id: {d['id']}, {e}")
                print_exc()
                # break

    if debug:
        for d in test_data:
            process_data(d)
    else:
        logger.warning(f"max_workers: {max_workers}")
        stats = run_thread_pool_cases(
            test_data,
            process_data,
            max_workers=max_workers,
            case_timeout_seconds=case_timeout_seconds,
            desc=f"Running {select_method}",
        )
        logger.info(
            f"All done. total={stats['total']}, success={stats['success']}, "
            f"timeouts={stats['timeouts']}, failed={stats['errors']}"
        )


if __name__ == "__main__":
    if is_vscode_debug_mode():
        logger.debug("Debug mode")
        select_method = "bfs"

        run_tree_search("kqapro", note="debug", debug=True, select_method=select_method)
    else:
        fire.Fire(run_tree_search)
"""
# grailqa
export actor_model=save-model-sft/small-GrailQA-v2-v1/Meta-Llama-3.1-8B-Instruct-epoch5
python tree_search.py --dataset grailqa --select_method bfs --note v1 --actor_model $actor_model
python tree_search.py --dataset grailqa --select_method dfs --note v1 --actor_model $actor_model
python tree_search.py --dataset grailqa --select_method randomwalk --note v1 --actor_model $actor_model
"""
