"""Shared state and operations for knowledge-base tree search."""

import math
import os
import pickle
import random
from dataclasses import dataclass, field
from time import time

from loguru import logger
from openai import OpenAI

from common.common_utils import colorful
from common.constant import ACTOR_KEY
from data_loader.load import get_qtype_key, make_system_prompt
from evaluation.runtime import (
    cal_metrics,
    clean_xsd,
    extract_dialog_last_valid_observation,
)
from kb.actions import parse_action, should_skip_observation

ACTOR_MODEL = "save-model-sft/small-KQAPro-v2/Meta-Llama-3.1-8B-Instruct-epoch5"
REWARD_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def wrap_text(text: str, max_len: int = 60) -> str:
    return "\n".join([text[i : i + max_len] for i in range(0, len(text), max_len)])


@dataclass
class Node:
    name: str
    content: str
    observation: str
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)

    # grouped by start node
    reward: float = 0.0
    # backprop
    total_reward: float = 0.0
    visit: int = 0

    # info
    is_terminal: bool = False
    is_exhausted: bool = False
    expand_fail_count: int = 0
    terminal_index: int = -1
    is_in_final_path: bool = False
    print = False
    create_time: float = field(default_factory=lambda: time())

    def __str__(self):
        text = colorful(f"Node: {self.name}") + "\n"
        text += f"{colorful('Thought', color='green')}: {self.thought}\n"
        text += f"{colorful('Action', color='green')}: {self.action}\n"
        text += f"{colorful('Observation', color='green')}: {self.observation}\n"
        text += f"{colorful('Visit', color='green')}: {self.visit}\n"
        text += f"{colorful('Reward', color='green')}: {self.reward}\n"
        text += f"{colorful('Total Reward', color='green')}: {self.total_reward}"
        return text

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "Node"):
        return self.name == other.name

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def thought(self) -> str:
        return self.content.split("\nAction:")[0].replace("Thought: ", "").strip()

    @property
    def action(self) -> str:
        if "\nAction:" in self.content:
            return self.content.split("\nAction:")[1].strip()
        else:
            return "No action."

    @property
    def message(self):
        """
        History:
        [
            {
                "role": "system",
                "content": "Task: xxx"
            },
            {
                "role": "user",
                "content": "Q: xxx"
            },
            {
                "role": "assistant",
                "content": "Thought: xxx.\nAction: xxx"
            },
            {
                "role": "user",
                "content": "Observation: xxx"
            },
            ......
        ]
        For regular nodes, it should be Thought + Action + Observation.
        """
        if self.is_root:
            return []

        assistant_part = {
            "role": "assistant",
            "content": f"Thought: {self.thought}.\nAction: {self.action}",
        }
        user_part = {"role": "user", "content": f"Observation: {self.observation}"}
        return [assistant_part, user_part]

    @property
    def path(self) -> list["Node"]:
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class _SearchUnpickler(pickle.Unpickler):
    """Resolve historical search class names in saved trees."""

    def find_class(self, module: str, name: str):
        """Load legacy pickle names through their current classes."""
        if module == "__main__" and name == "Node":
            return Node
        if module in {"__main__", "search.base", "search.fast_mcts"} and name in {
            "MCTSKBQA",
            "FastMCTSKBQA",
        }:
            from search.fast_mcts import FastMCTSKBQA

            return FastMCTSKBQA
        if module in {"__main__", "search.classic_mcts"} and name in {
            "MCTSKBQA_Classic",
            "ClassicMCTSKBQA",
        }:
            from search.classic_mcts import ClassicMCTSKBQA

            return ClassicMCTSKBQA
        return super().find_class(module, name)


class KBQATreeSearch:
    """Shared tree state, action expansion, and answer extraction for KBQA search."""

    def __init__(
        self,
        qid: str,
        question: str,
        golden_sparql: str,
        golden_answer: str,
        dataset: str,
        qtype: str | None = None,
        terminal_count_threshold: int = 1,
        actor_model: str = ACTOR_MODEL,
        reward_model: str = REWARD_MODEL,
        actor_n: int = 10,
        start_seed: int | None = None,
        use_constant_reward: bool = False,
        debug: bool = False,
    ):
        """Initialize the tree and actor client for one question."""
        self.qid = qid
        self.question = question
        self.golden_sparql = golden_sparql
        self.golden_answer = sorted(golden_answer)
        self.tooldesc = make_system_prompt(dataset, ft=True)
        self.dataset = dataset
        self.terminal_count = 0
        self.terminal_count_threshold = terminal_count_threshold
        self.qtype = qtype

        self.max_depth = 9 if dataset in ["webqsp", "cwq"] else 12
        self.max_preferred_depth = 5 if dataset in ["webqsp", "cwq"] else 9
        self.expand_failure_reward = -0.1
        self.expand_failure_threshold = 2

        self.f1s: list[float] = []
        self.valid_done_nodes: list[Node] = []

        self.actor_model = actor_model
        self.actor_n = actor_n
        self.reward_model = reward_model
        self.reward_temperature = 0.7
        self.reward_n = 10
        self.use_constant_reward = use_constant_reward

        self.debug = debug
        self.node_index = 0

        self._call_num = 0
        self._reward_num = 0
        self.start_seed = start_seed
        self.actor_client = self._build_actor_client()

        # human readable
        self.create_time = time()

        if dataset in ["webqsp", "cwq", "grailqa"]:
            self.db = "fb"
        elif dataset == "kqapro":
            self.db = "kqapro"
        else:
            raise ValueError(f"dataset: {dataset} not supported.")

        self.root = self._create_root_node()

    def __str__(self):
        text = colorful(self.__class__.__name__) + "\n"
        text += f"{colorful('qid', color='green')}: {self.qid}\n"
        text += f"{colorful('question', color='green')}: {self.question}\n"
        text += f"{colorful('golden_sparql', color='green')}: {self.golden_sparql}\n"
        text += f"{colorful('golden_answer', color='green')}: {self.golden_answer}\n"
        return text

    def constant_reward(self, *args, **kwargs):
        return 1

    def _configure_ig_reward(self):
        """Configure the scorer used by Fast MCTS and IG-Beam Search."""
        if self.use_constant_reward:
            self.reward_function = self.constant_reward
        else:
            from reward.conditional_prob_sglang import compute_information_gain

            self.reward_function = compute_information_gain

    def path_to_complete_messages(self, path: list[Node] | None = None) -> list[dict]:
        """Render a node path as the actor or reward model's conversation."""
        if "gemma" in self.actor_model:
            messages = [
                {
                    "role": "user",
                    "content": f"{self.tooldesc}\n\nQ: {self.question.strip()}",
                },
            ]
        else:
            messages = [
                {"role": "system", "content": self.tooldesc},
                {"role": "user", "content": f"Q: {self.question.strip()}"},
            ]

        for node in path or ():
            messages.extend(node.message)
        return messages

    def _create_root_node(self) -> Node:
        return Node(name="root", content=self.question, observation="No observation.")

    @staticmethod
    def _uses_official_openai(model_name: str) -> bool:
        return model_name.startswith(("gpt-", "o1-", "o3-", "o4-"))

    def _build_actor_client(self) -> OpenAI:
        if self._uses_official_openai(self.actor_model):
            return OpenAI(timeout=30)
        return OpenAI(
            base_url=os.environ.get("ACTOR_BASE_URL", "http://localhost:19000/v1"),
            api_key=ACTOR_KEY,
            timeout=30,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("actor_client", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.actor_client = self._build_actor_client()

    def select(self) -> Node:
        node = self.root
        while node.children:
            candidates = [
                n for n in node.children if not n.is_terminal and not n.is_exhausted
            ]
            if not candidates:
                return node
            node = max(candidates, key=lambda n: self._uct(n))
        return node

    def _uct(self, node: Node) -> float:
        """
        UCB1 (Upper Confidence Bound 1)
        UCB1 = Ave(X_i) + sqrt(2 * ln(n) / n_i)

        UCT (Upper Confidence bounds applied to Trees)
        UCT extends the UCB1 strategy to tree structures, suitable for multi-level complex decision problems.
        By recursively applying the UCB strategy at each node in the tree, it achieves efficient search and optimization across the entire decision space.

        UCT = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        Q(s,a): Average reward of node a, typically calculated as total_reward / visit
        """
        if node.visit == 0:
            return float("inf")

        c = math.sqrt(2)
        exploitation = node.total_reward / node.visit
        exploration = c * math.sqrt(math.log(node.parent.visit) / node.visit)
        uct = exploitation + exploration
        if self.debug:
            logger.debug(
                f"uct: {uct}. exploitation: {exploitation}, exploration: {exploration}"
            )
        return uct

    def _sample_candidates(
        self,
        node: Node,
        *,
        temperature: float,
        max_tokens: int,
        rollout: bool = False,
    ) -> tuple[list[dict], list[tuple[str, str]]]:
        """Sample actor actions, execute each distinct action, and keep distinct observations."""
        if node.is_terminal or node.is_exhausted:
            return [], []
        path = node.path
        if len(path) >= self.max_depth:
            node.is_exhausted = True
            if not rollout:
                self.backpropagate_with_reward(path, self.expand_failure_reward)
            return [], []

        messages = self.path_to_complete_messages(path)
        response = self.actor_client.chat.completions.create(
            model=self.actor_model,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            n=1 if rollout else self.actor_n,
            stop=["\nObservation", "\nThought"],
            max_tokens=max_tokens,
            seed=None if self.start_seed is None else self.start_seed + self._call_num,
        )
        self._call_num += 1

        visited_actions = set()
        visited_observations = set()
        candidates = []
        for choice in response.choices:
            if not choice.message or not choice.message.content:
                continue
            content = choice.message.content.split("Observation", 1)[0].strip()
            action = parse_action(content, execute=False, dataset=self.dataset)
            if should_skip_observation(action) or action in visited_actions:
                continue
            visited_actions.add(action)

            observation = parse_action(
                content, db=self.db, execute=True, dataset=self.dataset
            )
            if (
                should_skip_observation(observation)
                or observation in visited_observations
            ):
                continue
            visited_observations.add(observation)
            candidates.append((content, observation))

        if not rollout:
            depth = sum(message["role"] == "assistant" for message in messages)
            logger.info(f"Expanding {len(candidates)} children at depth {depth}.")
            if not candidates:
                node.expand_fail_count += 1
                self.backpropagate_with_reward(path, self.expand_failure_reward)
                if node.expand_fail_count >= self.expand_failure_threshold:
                    node.is_exhausted = True
        return messages, candidates

    @staticmethod
    def _resolve_done(
        content: str, observation: str, messages: list[dict], depth: int
    ) -> tuple[str, bool]:
        """Accept Done only after a usable SPARQL answer."""
        if "Action: Done" not in content:
            return observation, False

        last_action = messages[-2]["content"] if depth else ""
        if (
            "Action: " not in last_action
            or "ExecuteSPARQL" not in last_action.split("Action: ")[1]
        ):
            return (
                "Error: MUST use ExecuteSPARQL action to get the final answer.",
                False,
            )

        last_observation = messages[-1]["content"].split("Observation: ")[-1].strip()
        if last_observation == "[]":
            return (
                "Error: The final answer MUST NOT be empty, please rewrite the SPARQL query.",
                False,
            )
        if "Error: " in last_observation:
            return (
                "Error: The final answer is invalid, please rewrite the complete SPARQL query.",
                False,
            )
        if "pred:fact_" in last_observation:
            return (
                "Error: MUST NOT contain pred:fact_ in the final answer, please rewrite the complete SPARQL query.",
                False,
            )
        if "?e <" in last_observation:
            return (
                "Error: MUST NOT contain the variable ?e in the final answer, please rewrite the SPARQL query.",
                False,
            )
        return observation, True

    def _create_child(
        self,
        parent: Node,
        content: str,
        observation: str,
        *,
        is_terminal: bool,
        reward: float = 0.0,
    ) -> Node:
        """Build a child and retain the preceding SPARQL node for answer voting."""
        if is_terminal:
            self.terminal_count += 1
            self.valid_done_nodes.append(parent)
            parent.is_terminal = True
            logger.warning(f"Node {parent.name} is terminal.")

        child = Node(
            name=f"Node-{self.node_index}",
            content=content,
            observation=observation,
            parent=parent,
            reward=reward,
            is_terminal=is_terminal,
        )
        self.node_index += 1
        return child

    def expand_and_backpropagate(self, node: Node) -> None:
        """Expand, score, and backpropagate children without running a rollout."""
        parent_messages, candidates = self._sample_candidates(
            node, temperature=1, max_tokens=1024
        )
        depth = sum(message["role"] == "assistant" for message in parent_messages)
        for content, observation in candidates:
            observation, is_terminal = self._resolve_done(
                content, observation, parent_messages, depth
            )
            child_messages = [
                *parent_messages,
                {"role": "assistant", "content": content},
                {"role": "user", "content": f"Observation: {observation}"},
            ]
            reward = self.reward_function(
                messages=child_messages, q=self.question, model_name=self.reward_model
            )
            if not self.use_constant_reward:
                self._reward_num += 1

            child = self._create_child(
                node, content, observation, is_terminal=is_terminal, reward=reward
            )
            node.children.append(child)
            self.backpropagate_with_reward(child.path, reward)
            print(child)
            print()
            if is_terminal:
                break

    def backpropagate_with_reward(self, path: list[Node], reward: float):
        """
        Use depth penalty (linear decay) + discounted reward for backpropagation.
        Symbol definition:
            w: cumulative reward
            r: current reward
            γ: depth penalty coefficient, here is 0.1
            d_exp: expected maximum depth, here is 5
            d_e: current node depth
            δ: discount factor, here is 0.95
        Formula description:
            w = w + r * (1 - γ * max(0, d_e - d_exp))
            r = r * δ (apply discount when propagating upwards)
        """
        depth_penalty = 0.1  # depth penalty coefficient γ
        discount_factor = 0.95  # discount factor δ

        curr_depth = len(path) - 1

        # Start from the leaf node and propagate backwards
        for node in reversed(path):
            node.visit += 1

            # Apply depth penalty
            if curr_depth > self.max_preferred_depth:
                depth_penalty_value = depth_penalty * (
                    curr_depth - self.max_preferred_depth
                )
                adjusted_reward = reward * (1 - depth_penalty_value)
            else:
                adjusted_reward = reward

            # Accumulate adjusted reward
            node.total_reward += adjusted_reward

            # Apply discount factor
            reward = reward * discount_factor
            curr_depth -= 1

    def get_valid_done_paths(self, topk=None) -> list[list[Node]]:
        """Return valid answer paths in the order their terminal nodes were created."""
        nodes = sorted(self.valid_done_nodes, key=lambda node: node.create_time)
        return [node.path for node in nodes[:topk]]

    def cal_f1(self):
        best_paths = self.get_valid_done_paths()
        f1s = []
        for best_path in best_paths:
            messages = self.get_messages_from_path(best_path)
            prediction = extract_dialog_last_valid_observation({"messages": messages})
            golden_answers = clean_xsd(self.golden_answer)
            prediction = clean_xsd(prediction)
            _, _, average_f1, *_ = cal_metrics([golden_answers], [prediction])
            f1s.append(average_f1)
        self.f1s = f1s

    def _get_action_and_observation(self, path: list[Node]):
        messages = self.get_messages_from_path(path)
        return messages[-2]["content"], messages[-1]["content"]

    def _get_answer_dict_from_path(self, path: list[Node]):
        messages = self.get_messages_from_path(path)

        # for eval.
        messages.append({"role": "user", "content": "Stop condition detected."})

        qtype_key = get_qtype_key(self.dataset)
        d = {
            "id": self.qid,
            "question": self.question,
            "history": messages,
            "answers": self.golden_answer,
            "prediction": extract_dialog_last_valid_observation({"messages": messages}),
            qtype_key: self.qtype,
            "time_cost": self.end_time - self.create_time,
            "call_num": self._call_num,
            "reward_num": self._reward_num,
        }
        return d

    def get_answer_by_max_reward(self, _f1=False):
        """
        Required format for evaluation:
        - id
        - question
        - dialog
        - answers
        - prediction
        - [qtype]
            - for webqsp: infer_chain
            - for cwq: compositionality_type
            - for kqapro: qtype
        """
        valid_paths = self.get_valid_done_paths()
        if not valid_paths:
            if _f1:
                return "No valid path", 0
            return "No valid path"
        best_path = max(valid_paths, key=lambda p: p[-1].reward or 0)
        # set a special key for final answer
        for node in best_path:
            node.is_in_final_path = True
        answer = self._get_answer_dict_from_path(best_path)
        if _f1:
            average_f1 = self._cal_f1(answer)
            return answer, average_f1
        return answer

    def get_answer_by_obs_vote(self, _f1=False, topk=None):
        """count by the obs, return the path"""
        valid_paths = self.get_valid_done_paths(topk=topk)
        if not valid_paths:
            if _f1:
                return "No valid path", 0
            return "No valid path"

        obs_count = {}
        obs_to_path = {}

        # Count the number of occurrences of each observation and record the corresponding path
        for path in valid_paths:
            messages = self.get_messages_from_path(path)
            observation = messages[-1]["content"].split("Observation: ")[1].strip()
            obs_count[observation] = obs_count.get(observation, 0) + 1
            obs_to_path[observation] = path

        # Find the observation with the highest number of occurrences. If the most frequent observation has a count of 1, return get_answer_by_max_reward
        max_obs = max(obs_count.items(), key=lambda x: x[1])[0]
        max_count = obs_count[max_obs]
        if max_count == 1:
            return self.get_answer_by_max_reward(_f1=_f1)
            # return self.get_answer_by_llm_selection(_f1=_f1)

        best_path = obs_to_path[max_obs]

        # Set the final path mark
        for node in best_path:
            node.is_in_final_path = True

        # Get the answer
        answer_dict = self._get_answer_dict_from_path(best_path)
        if _f1:
            average_f1 = self._cal_f1(answer_dict)
            return answer_dict, average_f1
        return answer_dict

    def get_random_fallback_answer(self) -> dict:
        """Keep one random explored path when no valid Done path exists."""
        all_nodes: list[Node] = []

        # Walk the tree and collect nodes that actually executed SPARQL.
        def collect_nodes(node: Node):
            for child in node.children:
                all_nodes.append(child)
                collect_nodes(child)

        collect_nodes(self.root)
        executable_nodes = [
            node
            for node in all_nodes
            if "ExecuteSPARQL" in node.action
            and "Error: " not in node.observation
            and node.observation != "[]"
        ]

        # Prefer a SPARQL path the evaluator can parse; otherwise use any expanded node.
        candidates = executable_nodes or all_nodes
        if candidates:
            selected_node = random.Random(self.start_seed).choice(candidates)
            return self._get_answer_dict_from_path(selected_node.path)

        # Write a canonical empty prediction when the model produced no usable action.
        qtype_key = get_qtype_key(self.dataset)
        return {
            "id": self.qid,
            "question": self.question,
            "history": [
                {"role": "user", "content": f"Q: {self.question}"},
                {"role": "user", "content": "Stop condition detected."},
            ],
            "answers": self.golden_answer,
            "prediction": ["_None"],
            qtype_key: self.qtype,
            "time_cost": self.end_time - self.create_time,
            "call_num": self._call_num,
            "reward_num": self._reward_num,
        }

    def get_messages_from_path(self, path: list[Node]) -> list[dict]:
        assert path[0].is_root, "path[0] must be root"
        messages = []
        for node in path:
            if node.is_root:
                messages.append({"role": "user", "content": f"Q: {self.question}"})
            else:
                messages.append({"role": "assistant", "content": node.content})
                messages.append(
                    {"role": "user", "content": "Observation: " + node.observation}
                )
        return messages

    def _cal_f1(self, answer_dict: dict):
        prediction = clean_xsd(answer_dict["prediction"])
        golden_answers = clean_xsd(answer_dict["answers"])
        _precision, _recall, average_f1, *_ = cal_metrics(
            [golden_answers], [prediction]
        )
        return average_f1

    def save(self, save_dir: str):
        if save_dir.endswith("/"):
            save_dir = save_dir[:-1]
        # save_to_pkl(self, f"{save_dir}/{self.qid}.pkl")

    @staticmethod
    def load(file_path: str, errors: str = "strict") -> "KBQATreeSearch":
        """Load a saved search tree, including trees serialized under older class names."""
        with open(file_path, "rb") as stream:
            return _SearchUnpickler(stream, errors=errors).load()

    def _setprint_node(self):
        """
        Traverse, find the points of node.is_terminal and their paths, set print = True
        """
        terminal_nodes: list[Node] = []

        def _traverse(node: Node):
            if node.is_terminal:
                terminal_nodes.append(node)
            for child in node.children:
                _traverse(child)

        _traverse(self.root)

        # set print for terminal nodes and their paths
        for node in terminal_nodes:
            node.print = True
            while node.parent:
                node = node.parent

    def visualize_tree(
        self,
        format: str = "pdf",
        reduce_method: str = "vote",
        save_dir: str = "save-mcts-pdf",
    ) -> None:
        if reduce_method == "vote":
            answer, f1 = self.get_answer_by_obs_vote(_f1=True)
        elif reduce_method == "maxR":
            answer, f1 = self.get_answer_by_max_reward(_f1=True)
        else:
            raise ValueError(f"reduce_method: {reduce_method} not supported")

        self._setprint_node()

        from graphviz import Digraph

        dot = Digraph()

        max_len_line = 60

        # Info Node
        q = f"Q: {self.question}"
        golden_sparql = "Golden SPARQL: " + self.golden_sparql.replace("\n", " ")
        golden_answer = "Golden Answer: " + str(self.golden_answer).replace("\n", " ")
        label = (
            wrap_text(q, max_len_line)
            + "\n"
            + wrap_text(golden_sparql, max_len_line)
            + "\n"
            + wrap_text(golden_answer, max_len_line)
        )
        dot.node("info", label, shape="box", fontname="Helvetica", margin="0.2")

        def add_node_to_graph(node: Node, dot: Digraph):
            # if not node.print:
            #     return "Skip"

            node_id = str(id(node))

            thought = "Thought: " + node.thought
            action = "Action: " + node.action
            obs = "Observation: " + node.observation
            label = (
                node.name
                + "\n"
                + wrap_text(thought, max_len_line)
                + "\n\n"
                + wrap_text(action, max_len_line)
                + "\n\n"
                + wrap_text(obs, max_len_line)
                + "\n\n"
                + f"Score: {node.reward}"
            )

            if node.is_terminal and "Action: Done" in label:
                dot.node(
                    node_id,
                    label,
                    shape="box",
                    style="filled",
                    color="lightgreen",
                    fontname="Helvetica",
                    margin="0.1",
                )
            elif node.is_in_final_path:
                dot.node(
                    node_id,
                    label,
                    shape="box",
                    # style="filled",
                    # color="lightblue",
                    fontname="Helvetica",
                    margin="0.1",
                )

            else:
                dot.node(
                    node_id, label, shape="box", fontname="Helvetica", margin="0.1"
                )

            # add nodes and edges recursively
            for i, child in enumerate(node.children):
                child_id = str(id(child))
                res = add_node_to_graph(child, dot)
                if res != "Skip":
                    dot.edge(node_id, child_id)

        add_node_to_graph(self.root, dot)
        _f1s = "[" + "-".join([f"{f:.2f}" for f in self.f1s]) + "]"
        # {qtype}-{qid}-C{call num}-N{node num}-F{f1s}-B{f1}
        filename = f"{save_dir}/{self.qtype}-{self.qid}-C{self._call_num}-N{self.node_index}-F{_f1s}-B{f1}"
        dot.render(filename, format=format, cleanup=True)
