import ast
import os
import re
import time
from collections import Counter
from time import perf_counter

import httpx
from loguru import logger

from kb.client import get_actions_api

_TOOL_INFRA_MAX_ATTEMPTS = int(os.environ.get("KBQA_TOOL_INFRA_RETRY", "3"))
_TOOL_INFRA_RETRY_BACKOFF_S = float(os.environ.get("KBQA_TOOL_INFRA_BACKOFF", "1.0"))


class ToolInfraError(RuntimeError):
    """KB tool infra failure after retries; abort the current case."""


def is_tool_infra_error(obs: str | None) -> bool:
    if obs is None:
        return True
    text = str(obs).strip()
    if not text or text == "None":
        return True

    first_line = text.split("\n", 1)[0].strip()
    if first_line.startswith("# ") and first_line.endswith(" Error"):
        return True

    infra_markers = (
        "RetryError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ConnectError",
        "NetworkError",
        "ReadTimeout",
        "WriteTimeout",
        "RemoteProtocolError",
    )
    if any(marker in text for marker in infra_markers):
        return True
    if text.startswith("Error: timed out"):
        return True
    return False


def should_skip_observation(obs: str | None) -> bool:
    """Observations that must not become search/dialog nodes."""
    if obs is None:
        return True
    text = str(obs).strip()
    if not text or text == "None":
        return True
    if text == "[]":
        return True
    if is_tool_infra_error(text):
        return True
    if text.startswith("Error:") or text.startswith("Error!"):
        return True
    return False


def rm_duplicate_actions(content: str):
    """
    if multi \nAction: in content, only keep the first one.
    """
    if "\nAction:" in content:
        _cs = content.split("\nAction: ")
        content = "\nAction: ".join(_cs[:2])
    return content.strip()


def _execute_action_with_retry(action_str: str, db: str) -> str:
    """Parse literal arguments from model output and call a registered knowledge graph tool."""
    call = ast.parse(action_str, mode="eval").body
    tool_names = ("SearchNodes", "SearchTypes", "SearchGraphPatterns", "ExecuteSPARQL")
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id not in tool_names
        or any(keyword.arg is None for keyword in call.keywords)
    ):
        raise ValueError("Expected a supported tool call with literal arguments.")
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
    if len(kwargs) != len(call.keywords):
        raise ValueError("Duplicate tool argument.")

    last_obs = None
    last_exc = None
    for attempt in range(_TOOL_INFRA_MAX_ATTEMPTS):
        try:
            tool = get_actions_api(db)[tool_names.index(call.func.id)]
            obs = tool(*args, **kwargs)
            obs_str = "None" if obs is None else str(obs)
            if not is_tool_infra_error(obs_str):
                return obs_str
            last_obs = obs_str
            last_exc = None
        except httpx.HTTPError as exc:
            last_exc = exc
            last_obs = None
        except Exception as exc:
            if is_tool_infra_error(str(exc)) or exc.__class__.__name__ in {
                "ConnectError",
                "ConnectionError",
                "TimeoutError",
                "OSError",
            }:
                last_exc = exc
                last_obs = None
            else:
                raise

        if attempt + 1 < _TOOL_INFRA_MAX_ATTEMPTS:
            backoff = _TOOL_INFRA_RETRY_BACKOFF_S * (2**attempt)
            detail = last_obs or (
                f"{last_exc.__class__.__name__}: {last_exc}" if last_exc else "unknown"
            )
            logger.warning(
                f"Tool infra error, retry {attempt + 1}/"
                f"{_TOOL_INFRA_MAX_ATTEMPTS - 1} for {action_str[:200]}: {detail}"
            )
            time.sleep(backoff)

    detail = last_obs or (
        f"{last_exc.__class__.__name__}: {last_exc}" if last_exc else "unknown"
    )
    raise ToolInfraError(
        f"Tool infra error after {_TOOL_INFRA_MAX_ATTEMPTS} attempts "
        f"for {action_str}: {detail}"
    )


def validate_sparql_regex(action_str: str) -> str | None:
    """FILTER(...) requires type.object.name and @en in the same action."""
    if "FILTER" not in action_str:
        return None
    if "type.object.name" in action_str and "@en" in action_str:
        return None
    return (
        "Error: FILTER(REGEX/CONTAINS/LCASE) are used for entity "
        "disambiguation, and MUST be combined with type.object.name and @en to anchor "
        'the target entity (e.g., ?e ns:type.object.name "Entity"@en). '
        "Please rewrite the SPARQL query."
    )


def parse_action(text: str, db: str = None, execute=False, dataset: str = None):
    """
    Handling single quotes
        "Critics' Choice Movie Award"
        "Man of Steel's narrative location"
    """
    action_str = ""
    try:
        text = text.strip()

        action_names = [
            "SearchNodes",
            "SearchGraphPatterns",
            "ExecuteSPARQL",
            "Done",
        ]
        if dataset != "kqapro":
            action_names.insert(1, "SearchTypes")

        # if "Action:" not in text:
        #     return "Error: No valid action found. Please provide an action after `Action:` ."
        action_str = text.split("Action:")
        if len(action_str) < 2:
            return "Error: No valid action found. Please provide an action after `Action:` ."

        # find the start index of SearchNodes, SearchTypes, SearchGraphPatterns, ExecuteSPARQL, Done
        action_str = action_str[-1]
        if not any(i in action_str for i in action_names):
            if dataset == "kqapro" and "SearchTypes" in action_str:
                return (
                    "Error: SearchTypes is not available for KQAPro. "
                    "Use SearchNodes, SearchGraphPatterns, ExecuteSPARQL, or Done."
                )
            return "Error: Action invalid!"
        _s = [action_str.find(i) for i in action_names]
        _s = min([i for i in _s if i != -1])
        action_str = action_str[_s:].strip()
        if action_str.startswith("Done"):
            return "Done"

        # Handling single quotes
        if "\\'" not in action_str:
            action_str = action_str.replace("' ", "\\' ").replace("'s ", "\\'s ")
        # Handling \_
        action_str = action_str.replace("\\_", "_")

        # find the last ) as the end.
        action_str = action_str[: action_str.rfind(")") + 1]

        if ", type=" in action_str:
            action_str = action_str.split(", type=")[0] + ")"
        if ", timeout" in action_str:
            action_str = action_str.split(", timeout")[0] + ")"

        # if dataset == "grailqa":
        sparql_regex_err = validate_sparql_regex(action_str)
        if sparql_regex_err:
            return sparql_regex_err

        if execute:
            assert db is not None, "db must be provided when execute=True"
            start_time = perf_counter()
            try:
                obs_str = _execute_action_with_retry(action_str, db)
            finally:
                elapsed = perf_counter() - start_time
                logger.info(f"[action_time] {elapsed:.3f}s | {action_str}")
            return obs_str
        else:
            return action_str
    except ToolInfraError:
        raise
    except Exception as e:
        # print_exc()
        err = f"Error: Action parsing error. {e.__class__.__name__}: {str(e)}"
        logger.error(f"{err}. action_str: {action_str}. Raw: {text}")
        return err


def _is_valid_action(action_str):
    if not action_str or action_str.startswith("Done"):
        return False
    return not should_skip_observation(action_str)


def extract_predicates(sparql_query, db):
    assert db in ["kqapro", "fb"], "db must be one of [kqapro, fb]"
    if db == "kqapro":
        matches = re.compile(r"<([^\s]*?)>").findall(sparql_query)
        matches = [i for i in matches if not i.startswith("pred:")]
        for k in ["UNION", "LIMIT"]:
            if k in sparql_query:
                matches.append(k)

    elif db == "fb":
        matches = re.compile(r"ns:([^\s]*?) ").findall(sparql_query)
        matches = [i for i in matches if i and i != "type.object.name"]

    matches = "; ".join(sorted(set(matches), key=matches.index))
    return matches


def self_consistency_for_action(choices):
    """
    return: ranked (content, action)
    """
    actions = [parse_action(i, execute=False) for i in choices]
    valid_actions = [i for i in actions if _is_valid_action(i)]
    actions_counter = Counter(valid_actions).most_common()

    # debug
    # logger.debug(f"contents:")
    # print("\n".join(contents))
    # logger.debug(f"actions_counter:")
    # print("\n".join([str(i) for i in actions_counter]))

    # rank actions by actions_counter
    ranked_choicess = []
    for action, _count in actions_counter:
        for choice in choices:
            if action in choice:
                ranked_choicess.append(choice)
                break
    return ranked_choicess


if __name__ == "__main__":
    # python tool/action_execution.py
    x = """Action: SearchNodes("millennium anxiety")"""
    print(parse_action(x, db="fb", execute=True))
