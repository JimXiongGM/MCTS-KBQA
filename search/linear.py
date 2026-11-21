import os
import re
from glob import glob
from time import time

import fire
from loguru import logger
from openai import OpenAI

from common.common_utils import (
    DEFAULT_CASE_TIMEOUT_SECONDS,
    catch_keyboard_interrupt,
    colorful,
    is_vscode_debug_mode,
    run_thread_pool_cases,
    save_to_json,
)
from common.constant import ACTOR_KEY
from data_loader.load import get_qtype_key, load_test_data, make_system_prompt
from evaluation.runtime import extract_dialog_last_valid_observation
from kb.actions import ToolInfraError, parse_action, should_skip_observation

_DONE_WITHOUT_EXECUTE_SPARQL_HINT = (
    "Hint: You used Action: Done, but your previous action was not ExecuteSPARQL. "
    "You MUST run Action: ExecuteSPARQL(...) with a complete final SPARQL query "
    "to retrieve the final answer before using Action: Done."
)


def _extract_action_name(thought_action: str) -> str | None:
    match = re.search(r"Action:\s*(\w+)", thought_action)
    return match.group(1) if match else None


def _uses_official_openai(model_name: str) -> bool:
    return model_name.startswith(("gpt-", "o1-", "o3-", "o4-"))


def retry_on_empty_list(max_retries=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                if "seed" in kwargs:
                    kwargs["seed"] = kwargs["seed"] + attempt
                result = func(*args, **kwargs)
                if result is not None and "[]" not in str(result):
                    return result
                logger.warning(
                    f"{func.__name__} returned empty list, retry {attempt + 1}/{max_retries}"
                )
            return result

        return wrapper

    return decorator


@retry_on_empty_list()
def process_data(
    d: dict,
    db: str,
    model_name: str,
    client: OpenAI,
    save_dir: str,
    max_round_num: int = 15,
    system_prompt=None,
    seed=42,
):
    """
    different between apis:
    same:
        - messages, temperature, top_p, stop
    chatgpt:
        - max_tokens, n, presence_penalty, frequency_penalty
    llama:
        "max_new_tokens": 512,
        "do_sample": True,
        "repetition_penalty": 1,
        "num_return_sequences": 1,
    """
    assert "id" in d, "id must be provided."
    assert "question" in d, "question must be provided."
    assert save_dir is not None, "save_dir must be provided."

    q = d["question"].strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Q: {q}"},
    ]

    # hint again.
    if _uses_official_openai(model_name):
        messages.append(
            {
                "role": "user",
                "content": 'Now, use the format to answer the question: Thought: xxx.\nAction: xxx. ONLY ONE action per round, use Action: Done. to finish the question. use python code to execute the action, e.g. SearchNodes("...")',
            }
        )

    round_id = 0
    _last_out = ""
    last_executed_action = None

    # info
    start_time = time()
    call_num = 0

    # history: all inp and out
    history = []

    while round_id < max_round_num:
        if round_id > 0:
            logger.debug(f"round_id: {round_id}")

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stop=["\nObservation", "\nThought"],
            temperature=0.7,
            top_p=0.95,
            n=3,
            max_tokens=1024,
            seed=seed,
        )
        call_num += 1
        seed += 1

        # prepare choices
        choices = [
            choice.message.content
            for choice in response.choices
            if choice.message and choice.message.content
        ]
        choices = [c.split("Observation")[0] for c in choices if c]

        history.append({"round_id": round_id, "choices": choices})

        # Attempt to execute the first valid action in the actions.
        thought_action = choices[0].strip() if choices else ""
        Observation = None
        try:
            if thought_action:
                Observation = parse_action(thought_action, db=db, execute=True)
            for content in choices:
                _obs = parse_action(content, db=db, execute=True)
                if _obs and not should_skip_observation(_obs):
                    Observation = _obs
                    thought_action = content.strip()
                    break
        except ToolInfraError as e:
            logger.error(f"Tool infra error, abort case {d['id']}: {e}")
            d["tool_error"] = str(e)
            d["history"] = messages
            d["model_name"] = model_name
            d["time_cost"] = time() - start_time
            d["call_num"] = call_num
            save_to_json(d, f"{save_dir}/{d['id']}.json")
            return

        if Observation is None:
            return
        Observation = str(Observation).strip()
        current_action = _extract_action_name(thought_action)

        if Observation == "Done" and last_executed_action != "ExecuteSPARQL":
            if not thought_action.startswith("Thought: "):
                thought_action = "Thought: " + thought_action
            messages.append({"role": "assistant", "content": thought_action})
            messages.append(
                {"role": "user", "content": _DONE_WITHOUT_EXECUTE_SPARQL_HINT}
            )
            round_id += 1
            continue

        if current_action and current_action != "Done":
            last_executed_action = current_action

        if thought_action == _last_out:
            messages.append({"role": "user", "content": "STOP because of repetition."})
            break
        _last_out = thought_action

        if not thought_action.startswith("Thought: "):
            thought_action = "Thought: " + thought_action
        messages.append({"role": "assistant", "content": thought_action})

        # debug
        print()
        print(colorful("LLM: ", color="yellow"), end="")
        print(thought_action.replace("\n", "\\n"))
        print(colorful("Observation: ", color="yellow"), end="")
        print(Observation)

        if Observation != "Done":
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: " + Observation,
                }
            )
            round_id += 1
        else:
            messages.append({"role": "user", "content": "Stop condition detected."})
            break

        # debug
        # break

    d["history"] = messages

    # add model info
    d["model_name"] = model_name
    d["time_cost"] = time() - start_time
    d["call_num"] = call_num
    d["prediction"] = extract_dialog_last_valid_observation({"messages": messages})

    save_to_json(d, f"{save_dir}/{d['id']}.json")
    return d["prediction"]


@catch_keyboard_interrupt
def run(
    dataset,
    model_name,
    debug=False,
    num_each_type=None,
    note="",
    seed=42,
    max_workers=12,
    max_round_num=15,
    case_timeout_seconds=DEFAULT_CASE_TIMEOUT_SECONDS,
    full=False,
    use_sample=False,
    dataset_base_dir=None,
    save_root="save-fast-mcts-kbqa/linear",
    save_dir=None,
):
    env_use_sample = os.environ.get("USE_SAMPLE")
    if env_use_sample is not None:
        use_sample = env_use_sample.lower() in ("true", "1", "yes")
        logger.info(f"USE_SAMPLE={env_use_sample!r} -> use_sample={use_sample}")

    # NOTE: only use test set!!
    test_data = load_test_data(
        dataset,
        num_each_type=num_each_type,
        full=full,
        dataset_base_dir=dataset_base_dir,
        use_sample=use_sample,
    )

    # random sample 100 items
    # import random
    # random.seed(123)
    # random.shuffle(test_data)
    # test_data = test_data[:50]

    logger.info(f"len(test_data): {len(test_data)}")

    _setting = model_name.split("/")[-1] + f"-seed{seed}"
    if note:
        _setting = _setting + f"-{note}"

    # save to: save-fast-mcts-kbqa/linear/{model}-seed{seed}/{dataset}/{id}.json
    save_dir = save_dir or f"{save_root}/{_setting}/{dataset}"
    logger.info(f"saving to: {save_dir}")

    skip_ids = set()
    if not debug:
        for p in glob(f"{save_dir}/*.json"):
            skip_ids.add(os.path.basename(p).split(".json")[0])

    logger.info(f"Skip id: {len(skip_ids)}")
    test_data = [d for d in test_data if str(d["id"]) not in skip_ids]
    logger.warning(f"Remaining test test_data: {len(test_data)}")

    db = "fb" if dataset in ["webqsp", "cwq", "grailqa"] else "kqapro"
    ft = not _uses_official_openai(model_name)
    if _uses_official_openai(model_name):
        client = OpenAI(timeout=30)
    else:
        client = OpenAI(
            base_url=os.environ.get("ACTOR_BASE_URL", "http://127.0.0.1:19000/v1"),
            api_key=ACTOR_KEY,
            timeout=30,
        )

    def build_system_prompt(item: dict) -> str:
        if ft:
            return make_system_prompt(dataset, ft=True)
        if dataset in ("cwq", "kqapro"):
            qtype = item[get_qtype_key(dataset)]
            return make_system_prompt(dataset, ft=False, qtype=qtype)
        else:
            return make_system_prompt(dataset, ft=False)

    results = []

    if debug:
        for idx, item in enumerate(test_data, start=1):

            # if str(item["id"]) not in ["2100289003000"]:
            #     continue

            result = process_data(
                item,
                db=db,
                model_name=model_name,
                client=client,
                save_dir=save_dir,
                system_prompt=build_system_prompt(item),
                seed=seed,
                max_round_num=max_round_num,
            )
            if result is not None:
                results.append(result)
        return results

    start_time = time()

    def _process(item):
        return process_data(
            item,
            db=db,
            model_name=model_name,
            client=client,
            save_dir=save_dir,
            system_prompt=build_system_prompt(item),
            seed=seed,
            max_round_num=max_round_num,
        )

    results, stats = run_thread_pool_cases(
        test_data,
        _process,
        max_workers=max_workers,
        case_timeout_seconds=case_timeout_seconds,
        desc="interactive-kbqa",
        collect_results=True,
    )

    logger.info(
        f"All done. total={stats['total']}, success={stats['success']}, "
        f"timeouts={stats['timeouts']}, failed={stats['errors']}, "
        f"time_cost={time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    # model_name = "save-model-sft/small-WebQSP-CWQ/Meta-Llama-3.1-8B-Instruct-epoch3"
    # model_name = "save-model-sft/small-KQAPro-v2/Meta-Llama-3.1-8B-Instruct-epoch5"
    # run("cwq", "meta-llama/Meta-Llama-3.1-8B-Instruct", debug=1, num_each_type=100, qtype="superlative")
    # run(dataset="webqsp", model_name=model_name, debug=1, num_each_type=100)

    if is_vscode_debug_mode():
        logger.debug("Debug mode")
        dataset = "webqsp"
        model_name = "gpt-4.1-2025-04-14"
        run(
            dataset=dataset,
            model_name=model_name,
            debug=True,
        )
    else:
        fire.Fire(run)

"""
# num_each_type: Only run the first num_each_type cases.
export actor_model=save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5
export dataset_base_dir=data

# ---------- Linear (save-fast-mcts-kbqa/linear) ---------- #
# GPT4.1 -> save-fast-mcts-kbqa/linear/gpt-4.1-2025-04-14-seed42/{dataset}/{id}.json
python interactive_kbqa.py --dataset webqsp --model_name gpt-4.1-2025-04-14 --seed 42 --dataset_base_dir ${dataset_base_dir}
python interactive_kbqa.py --dataset cwq --model_name gpt-4.1-2025-04-14 --seed 42 --dataset_base_dir ${dataset_base_dir}

# SFT -> save-fast-mcts-kbqa/linear/Meta-Llama-3.1-8B-Instruct-epoch5-seed{seed}/{dataset}/{id}.json
python interactive_kbqa.py --dataset webqsp --model_name ${actor_model} --seed 1 --dataset_base_dir ${dataset_base_dir}
python interactive_kbqa.py --dataset cwq --model_name ${actor_model} --seed 1 --dataset_base_dir ${dataset_base_dir}

# major vote: 5 seeds, then merge
for seed in 1 10 100 1000 10000; do
    python interactive_kbqa.py --dataset webqsp --model_name ${actor_model} --seed ${seed} --dataset_base_dir ${dataset_base_dir}
done
python -m evaluation.linear_vote \
    --path save-fast-mcts-kbqa/linear/Meta-Llama-3.1-8B-Instruct-epoch5-seed1 \
    --dataset webqsp \
    --output_path save-fast-mcts-kbqa/linear/Meta-Llama-3.1-8B-Instruct-epoch5-major-vote/webqsp

--------------------------------
# Run the released WebQSP evaluation set.
python -m search.linear \
  --dataset webqsp \
  --model_name gpt-4.1-2025-04-14 \
  --seed 42 \
  --max_workers 4 \
  --case_timeout_seconds 600 \
  --dataset_base_dir "data" \
  --save_dir save-fast-mcts-kbqa/linear/gpt-4.1-2025-04-14-seed42/webqsp

# Run the released CWQ evaluation set.
python -m search.linear \
  --dataset cwq \
  --model_name gpt-4.1-2025-04-14 \
  --seed 42 \
  --max_workers 4 \
  --case_timeout_seconds 600 \
  --dataset_base_dir "data" \
  --save_dir save-fast-mcts-kbqa/linear/gpt-4.1-2025-04-14-seed42/cwq
"""
