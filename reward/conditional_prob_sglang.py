"""
Same reward logic as condi_prob_from_vllm, but uses SGLang native /generate with
logprob_start_len=0 to get full prompt logprobs (SGLang chat completions only return
logprobs for generated tokens).
"""

import os
from functools import lru_cache

import numpy as np
import requests
from loguru import logger
from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer used by the reward service from the local model directory."""
    return AutoTokenizer.from_pretrained(model_name, local_files_only=True)


@lru_cache(maxsize=128)
def get_sub_tokens(text: str, model_name: str) -> list[str]:
    tokenizer = get_tokenizer(model_name)
    return tokenizer.tokenize(text)


def find_sublist(lst, sub):
    lst = [x.replace("Ġ", "").strip() for x in lst]
    sub = [x.replace("Ġ", "").strip() for x in sub]
    for i in range(len(lst) - len(sub) + 1):
        if lst[i : i + len(sub)] == sub:
            return i
    return -1


# Placeholder for missing logprob (e.g. BOS); must be excluded from PPL to avoid inf
_INVALID_LOGPROB = -1e9
# Normal logprobs are in ~[-20, 0]; exclude placeholder and absurd values
_MIN_VALID_LOGPROB = -50.0


def calc_ppl(logprobs_slice):
    valid = [lp for lp in logprobs_slice if _MIN_VALID_LOGPROB <= lp <= 0]
    if not valid:
        return float("inf")
    nll = -np.sum(valid)
    return np.exp(nll / len(valid))


def _messages_to_prompt(messages_with_q: list[dict], model_name: str) -> str:
    tokenizer = get_tokenizer(model_name)
    out = tokenizer.apply_chat_template(
        messages_with_q,
        tokenize=False,
        add_generation_prompt=True,
        tokenizer=tokenizer,
    )
    if isinstance(out, list):
        return "".join(str(x) for x in out)
    return str(out)


def compute_information_gain(
    messages,
    q,
    text="the original question is:",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    k: float = 10,
    debug=False,
    url: str | None = None,
    api_key: str = "kbqa",
):
    assert len(messages) > 0, "messages is empty"
    assert messages[-1]["role"] == "user", "Last action is not user"
    # Allow each dataset to use its own reward server when runs are parallel.
    url = url or os.environ.get("REWARD_BASE_URL", "http://127.0.0.1:29000")

    actions = []
    last_observation = None
    for m in messages:
        if m["role"] == "system":
            continue
        elif m["content"].startswith("Q:"):
            continue
        elif m["role"] == "assistant":
            action = (
                m["content"].split("Action:")[1].strip()
                if "Action:" in m["content"]
                else "No Action"
            )
            if "SearchNodes" in action:
                action = "SearchNodes"
            elif "SearchGraphPatterns" in action:
                action = action.split(", semantic")[0] + ")"
            actions.append("Action: " + action)
        else:
            last_observation = m["content"]

    sys_prompt = """You are an AI assistant. I will provide you with a sequence of actions and observations from a knowledge base query session. Your task is to reconstruct the original question that would have led to these actions and observations.""".strip()
    history = actions.copy()
    if last_observation is not None:
        history.append(last_observation)
    history_str = "\n".join(history)

    if "gemma" in model_name:
        messages_with_q = [
            {"role": "user", "content": f"{sys_prompt}\n\n{history_str}"}
        ]
    else:
        messages_with_q = [{"role": "system", "content": sys_prompt}]
        messages_with_q.append({"role": "user", "content": history_str})

    messages_with_q.append(
        {
            "role": "assistant",
            "content": f"Based on this interaction history, the original question is: {q}",
        }
    )

    prompt_text = _messages_to_prompt(messages_with_q, model_name)

    response = requests.post(
        url.rstrip("/") + "/generate",
        json={
            "text": prompt_text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "return_text_in_logprobs": True,
        },
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Handle both single dict and list of one dict (router may return list)
    if isinstance(data, list):
        if not data:
            raise RuntimeError("SGLang /generate returned empty list")
        data = data[0]

    meta = data.get("meta_info") or {}
    input_logprobs = meta.get("input_token_logprobs") or []
    # Each item: [logprob, token_id, token_text]; logprob or token_text can be None
    tokens = []
    logprobs = []
    for item in input_logprobs:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        logprob, token_id, token_text = (
            item[0],
            item[1],
            item[2] if len(item) > 2 else None,
        )
        if token_text is None and token_id is not None:
            try:
                token_text = get_tokenizer(model_name).decode([token_id])
            except Exception:
                token_text = ""
        tokens.append(token_text or "")
        logprobs.append(logprob if logprob is not None else _INVALID_LOGPROB)

    if not tokens or len(logprobs) != len(tokens):
        raise RuntimeError(
            "SGLang response has no or invalid input_token_logprobs. "
            "Ensure server supports return_logprob and logprob_start_len=0."
        )

    sub_tokens = get_sub_tokens(text, model_name)[1:]
    start_idx = find_sublist(tokens, sub_tokens)
    if start_idx < 0:
        logger.warning(
            f"Could not find '{text}' in token sequence (got {len(tokens)} tokens). Using neutral score 1.0."
        )
        score = 1.0
        last_observation = messages[-1]["content"]
        if "Error: " in last_observation or " []" in last_observation:
            score = score * 0.85
        if debug:
            print()
            logger.debug(f"Question: {q} (fallback score={score})")
            print()
        return score

    q_start_idx = start_idx + len(sub_tokens) - 1
    ppl_with_no_q = calc_ppl(logprobs[:q_start_idx])
    ppl_with_q = calc_ppl(logprobs)
    if (
        not np.isfinite(ppl_with_no_q)
        or not np.isfinite(ppl_with_q)
        or ppl_with_no_q <= 0
        or ppl_with_q <= 0
    ):
        logger.warning(
            f"Invalid PPL (no_q={ppl_with_no_q}, with_q={ppl_with_q}), using neutral score 1.0"
        )
        log_prob_ratio = float("nan")
        score = 1.0
    else:
        log_prob_ratio = np.log(ppl_with_no_q) - np.log(ppl_with_q)
        score = 1 / (1 + np.exp(-k * log_prob_ratio))
        score = score / 0.5

    last_observation = messages[-1]["content"]
    if "Error: " in last_observation or " []" in last_observation:
        score = score * 0.85

    if debug:
        print()
        logger.debug(f"Question: {q}")
        actions = [
            m["content"].split("Action: ")[1].strip()
            for m in messages
            if m["role"] == "assistant"
        ]
        logger.debug("Actions:\n" + "\n".join(actions))
        logger.debug(
            f"PPL no Q: {ppl_with_no_q}, PPL with Q: {ppl_with_q}, Log probability ratio: {log_prob_ratio}, Score: {score}"
        )
        print()

    return score


def test_1():
    """Minimal test: same reward logic as condi_prob_from_vllm, via SGLang /generate."""
    q = "what new movies is robert pattinson in?"
    messages_0 = [
        {"role": "system", "content": "what new movies is robert pattinson in?"},
        {
            "role": "assistant",
            "content": 'Thought: ...\nAction: SearchNodes("Robert Pattinson")',
        },
        {
            "role": "user",
            "content": 'Observation: ["Robert Pattinson" | Description: ...]',
        },
    ]
    compute_information_gain(messages_0, q, debug=True)


if __name__ == "__main__":
    # python reward_model/conditional_prob_sglang.py
    # Requires SGLang server on 29000 with model (e.g. meta-llama/Meta-Llama-3.1-8B-Instruct)
    test_1()
