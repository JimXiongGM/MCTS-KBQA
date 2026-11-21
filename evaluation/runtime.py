import ast
import re
import string
from collections import defaultdict
from copy import deepcopy
from datetime import date
from glob import glob
from typing import Dict, List

import regex
from loguru import logger

from common.common_utils import colorful, read_json, underline_text
from common.path_utils import (
    get_dataset_split_glob,
    load_kqapro_test_ids,
    load_sample_ids,
)
from evaluation.webqsp import CalculatePRF1, cal_metrics
from kb.actions import should_skip_observation


def clean_obs(result: str):
    """
    imi', 'Pasenge..." -> find last ', ' and remove the last.
    """
    if result.endswith("..."):
        idx = result.rfind("', '")
        result = result[:idx] + "']"
    return result


def clean_xsd(item):
    """
    True -> ['True']
    [True] -> ['True']
    ['"2004"^^xsd:gYear'] -> ['2004']
    """
    if isinstance(item, list) or isinstance(item, set):
        item = sorted(set([str(clean_xsd(i)) for i in item]))
    elif isinstance(item, str) and "^^xsd:" in item:
        item = item.split("^^xsd:")[0].strip('"')
        try:
            item = ast.literal_eval(item)
            item = str(round(item, 4))
        except:
            pass
    elif isinstance(item, bool):
        item = str(item)
    elif item is None:
        item = "_None"
    return item


def _normalize_dialog(d: Dict) -> List[dict]:
    if not d:
        return []
    history = d.get("history")
    if history is None:
        history = d.get("messages")
    if not history:
        return []

    entries = []
    for item in history:
        if isinstance(item, dict):
            entries.append(
                {
                    "role": item.get("role", ""),
                    "content": item.get("content", ""),
                }
            )
            continue

        content = str(item)
        if content.startswith("Observation:") or content == "Stop condition detected.":
            role = "user"
        elif content.startswith("Thought:") or "Action:" in content:
            role = "assistant"
        else:
            role = "user"
        entries.append({"role": role, "content": content})
    return entries


def _is_execute_sparql_action(content: str) -> bool:
    return "ExecuteSPARQL" in content and "Action:" in content


def _is_execute_sparql_observation(content: str) -> bool:
    if not content.startswith("Observation: "):
        return False
    obs_body = content[len("Observation: ") :]
    return (
        " | " not in content
        and not should_skip_observation(obs_body)
        and "?e " not in content
        and "pred:fact_" not in content
    )


def _parse_observation_literal(obs_text: str):
    obs_text = obs_text.strip()
    if obs_text.startswith("Observation: "):
        obs_text = obs_text[len("Observation: ") :]
    obs_text = obs_text.replace("\nThought:", "").strip()
    obs_text = clean_obs(obs_text)

    if obs_text in ("True", "False"):
        return obs_text == "True"
    if obs_text == "[]":
        return []

    try:
        return ast.literal_eval(obs_text)
    except (SyntaxError, ValueError):
        pass

    if obs_text.startswith("["):
        items = []
        for match in re.finditer(
            r"'((?:\\'|[^'])*)'|\"((?:\\\"|[^\"])*)\"",
            obs_text,
        ):
            value = match.group(1) if match.group(1) is not None else match.group(2)
            items.append(value.replace("\\'", "'").replace('\\"', '"'))
        return items

    return None


def _normalize_execute_sparql_answer(value):
    if value is None:
        return ["_None"]
    if isinstance(value, bool):
        return [value]
    if isinstance(value, list):
        if not value:
            return ["_None"]
        if isinstance(value[0], tuple):
            answers = []
            for item in value:
                if len(item) < 2:
                    continue
                ans = item[1]
                if isinstance(ans, str):
                    white_idx = ans.find(" ")
                    if white_idx != -1:
                        ans = ans[white_idx + 1 :]
                    ans = ans.replace("@en", "").strip('"')
                answers.append(str(ans))
            return sorted(set(answers)) if answers else ["_None"]
        answers = [clean_xsd(i) for i in value]
        return sorted(set(answers), key=answers.index) if answers else ["_None"]
    return ["_None"]


def _find_last_execute_sparql_observation(entries: List[dict]):
    for index in range(len(entries) - 2, -1, -1):
        action = entries[index]["content"]
        observation = entries[index + 1]["content"]
        if not _is_execute_sparql_action(action):
            continue
        if not _is_execute_sparql_observation(observation):
            continue
        return index + 1, observation
    return None, None


def extract_dialog_last_observation(d):
    return extract_dialog_last_valid_observation(d)


def _is_digit(s):
    try:
        s = float(s)
        return True
    except ValueError:
        return False


def _clean_number_and_str(pred_ans: List[str]):
    """
    rm numeric item if str item exists
    e.g.
        ["2", "usa"] -> ["usa"]
    """
    if len(pred_ans) > 1:
        if any([_is_digit(i) for i in pred_ans]):
            if any([not _is_digit(i) for i in pred_ans]):
                pred_ans = [i for i in pred_ans if not _is_digit(i)]
    return pred_ans


def is_valid_observation(content: str):
    return _is_execute_sparql_observation(content)


def extract_dialog_last_valid_observation(d: Dict, return_index=False):
    if not d:
        return None

    entries = _normalize_dialog(d)
    if not entries:
        return ["_None"]

    max_index, observation = _find_last_execute_sparql_observation(entries)
    if observation is None:
        pred_ans = ["_None"]
    else:
        parsed = _parse_observation_literal(observation)
        pred_ans = _normalize_execute_sparql_answer(parsed)

    pred_ans = _clean_number_and_str(pred_ans)
    if return_index:
        return pred_ans, max_index or 0
    return pred_ans


def extract_dialog_last_valid_observation_kqa(d):
    if not d:
        return None
    pred_ans = []
    for item in d["history"][::-1]:
        if (
            item["role"] == "user"
            and item["content"].startswith("Observation: ")
            and " | " not in item["content"]
        ):
            obs = item["content"].replace("Observation: ", "")
            if "?e " in obs:
                continue
            obs = clean_obs(obs)
            try:
                obs = eval(obs)
            except (SyntaxError, NameError):
                continue

            # "Observation: [('Bernalillo County', '3027.0')]" -> ['Bernalillo County', '3027.0']
            if isinstance(obs, list) and len(obs) == 1 and isinstance(obs[0], tuple):
                pred_ans = list(obs[0])
                break

            # for kqa
            if "pred:fact_r" in str(obs) and len(obs) > 3:
                # rm pred:fact*; replace statement_is_subject_of to zzz;
                obs = [tup for tup in obs if "pred:fact" not in tup[0]]
                for i in range(len(obs)):
                    if obs[i][0] == "statement_is_subject_of":
                        obs[i] = ("zzz_statement_is_subject_of", obs[i][1])
                # sort by the first element
                obs = sorted(obs, key=lambda x: x[0])[0]
                pred_ans = [obs[1]]
                break
            elif (
                isinstance(obs, list) and len(obs) > 0 and not isinstance(obs[0], tuple)
            ):
                pred_ans = [clean_xsd(i) for i in obs]
                break
            elif isinstance(obs, bool):
                pred_ans = [obs]
                break
    pred_ans = sorted(set(pred_ans)) if len(pred_ans) > 0 else ["_None"]
    pred_ans = _clean_number_and_str(pred_ans)
    return pred_ans


def cal_dialog_info(data):
    """
    return:
        ave_turn: average all dialog turns
        success_rate: end with "Stop condition detected."
        ave_turn_done: average dialog turns for success dialog
    """
    # exclude the first "You are an AI assistant."
    ave_turn = sum([len(d["history"]) - 1 for d in data]) / len(data)
    success_rate = sum(
        [d["history"][-1]["content"] == "Stop condition detected." for d in data]
    ) / len(data)
    num_success = sum(
        [d["history"][-1]["content"] == "Stop condition detected." for d in data]
    )
    ave_turn_done = (
        sum(
            [
                len(d["history"]) - 1
                for d in data
                if d["history"][-1]["content"] == "Stop condition detected."
            ]
        )
        / num_success
    )
    return round(ave_turn, 2), round(success_rate, 2), round(ave_turn_done, 2)


def evaluation_webqsp(
    save_dir,
    full=False,
    force_full=False,
    use_sample=False,
    dataset_base_dir=None,
):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    # small data
    small_data_ids = []
    if use_sample:
        small_data_ids = load_sample_ids("webqsp", dataset_base_dir)
    elif not full:
        paths = glob(
            get_dataset_split_glob("webqsp", dataset_base_dir=dataset_base_dir)
        )
        for p in paths:
            _sdata = read_json(p)
            small_data_ids.extend([d["id"] for d in _sdata])
        small_data_ids = set(small_data_ids)

    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        try:
            d = read_json(p)
            if small_data_ids and d["id"] not in small_data_ids:
                continue
            d["prediction"] = extract_dialog_last_valid_observation(d)
            data.append(d)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

    # [("type", data), ...]
    _data1 = [d for d in data if len(d["infer_chain"]) == 1]
    _data2 = [d for d in data if len(d["infer_chain"]) == 2]

    # if force_full, make sure each type has 150 data.
    if force_full:
        bad_item = {
            "history": [{"role": "user", "content": "None"}],
            "answers": ["_None"],
            "prediction": ["None"],
        }
        if len(_data1) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data1)} empty items to chain_len_1"
            )
            _data1 += [bad_item] * (150 - len(_data1))
        if len(_data2) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data2)} empty items to chain_len_2"
            )
            _data2 += [bad_item] * (150 - len(_data2))
        data = _data1 + _data2

    tp_data = [
        ("chain_len_1", _data1),
        ("chain_len_2", _data2),
        ("total      ", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # cal metrics for webqsp and cwq; cal acc for kqapro and metaqa
        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\tRandomHit"
            if data and "history" in data[0]:
                # head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
                head = head + "\tcall_num\ttime_cost"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t[{average_f1}]\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{average_random_hit}"

        # turn info
        # if data and "history" in data[0]:
        #     ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
        #     line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        # call info
        if data and "call_num" in data[0]:
            _data = [d for d in data if "call_num" in d]
            call_num = sum([d["call_num"] for d in _data]) / len(_data)
            time_cost = sum([d["time_cost"] for d in _data]) / len(_data)
            line = line + f"\t{call_num:.2f}\t{time_cost:.2f}"

        print(line)
    print()
    return line


def _clean_latin_encoding(s: str | List[str]):
    if isinstance(s, str):
        try:
            return s.encode("latin-1").decode("utf-8")
        except:
            return s
    elif isinstance(s, list):
        return [_clean_latin_encoding(i) for i in s]
    else:
        raise ValueError(f"Invalid type: {type(s)}")


def _clean_float(pred_ans: List[str]):
    # If len(pred_ans) == 2 with one number and one string, return the string
    if len(pred_ans) == 2:
        try:
            float(pred_ans[0])
            print(f"clean_float: {pred_ans[0]} -> {pred_ans[1]}")
            return [pred_ans[1]]
        except:
            pass

        try:
            float(pred_ans[1])
            print(f"clean_float: {pred_ans[1]} -> {pred_ans[0]}")
            return [pred_ans[0]]
        except:
            pass
    return pred_ans


def evaluation_grailqa(
    save_dir, full=False, force_full=False, use_sample=False, dataset_base_dir=None
):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    # small data
    small_data_ids = []
    if use_sample:
        small_data_ids = load_sample_ids("grailqa", dataset_base_dir)
    elif not full:
        paths = glob(
            get_dataset_split_glob("grailqa", dataset_base_dir=dataset_base_dir)
        )
        for p in paths:
            _sdata = read_json(p)
            small_data_ids.extend(str(d["id"]) for d in _sdata)
        small_data_ids = set(small_data_ids)

    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        try:
            d = read_json(p)
            if (use_sample or not full) and str(d["id"]) not in small_data_ids:
                continue
            d["prediction"] = _clean_float(extract_dialog_last_valid_observation(d))
            data.append(d)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue

    # [("type", data), ...]
    _data1 = [d for d in data if d["level"] == "i.i.d."]
    _data2 = [d for d in data if d["level"] == "compositional"]
    _data3 = [d for d in data if d["level"] == "zero-shot"]

    # if force_full, make sure total data = 600.
    if force_full:
        if len(data) < 600:
            logger.warning(
                f"Full data Mode! Add {600 - len(data)} empty items to total"
            )
            data += [
                {
                    "history": [{"role": "user", "content": "None"}],
                    "answers": ["_None"],
                    "prediction": ["None"],
                }
            ] * (600 - len(data))

    tp_data = [
        ("i.i.d.", _data1),
        ("compositional", _data2),
        ("zero-shot", _data3),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data_list in tp_data:
        print(f"{tp}: {len(data_list)}", end="\t")
    print()

    head = (
        f"{'type':<13}\t{'precision':>5}\t{'recall':>5}\t{'ave_f1':>5}\t"
        f"{'f1_ave':>5}\t{'accuracy':>5}\t{'hit@1':>5}\t{'hit@5':>5}\t"
        f"{'hit@10':>5}\t{'RandomHit':>5}\t"
    )
    if data and "history" in data[0]:
        # head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
        head = head + f"{'call_num':>5}\t{'time_cost':>5}"
    print(colorful(head, "red"))

    for idx, (tp, data_list) in enumerate(tp_data):
        if not data_list:
            continue
        ans_key = "answer" if "answer" in data_list[0] else "answers"
        golden_answers = [clean_xsd(i[ans_key]) for i in data_list]
        predictions = [clean_xsd(i["prediction"]) for i in data_list]

        # clean latin encoding
        golden_answers = _clean_latin_encoding(golden_answers)
        predictions = _clean_latin_encoding(predictions)

        # cal metrics for webqsp and cwq; cal acc for kqapro and metaqa
        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)

        line = (
            f"{tp:<13}\t{precision:>5.2f}\t{recall:>5.2f}\t[{average_f1:>5.2f}]\t"
            f"{f1_average:>5.2f}\t{accuracy:>5.2f}\t{hit1:>5.2f}\t{hit5:>5.2f}\t"
            f"{hit10:>5.2f}\t{average_random_hit:>5.2f}\t"
        )

        # turn info
        # if data and "history" in data[0]:
        #     ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
        #     line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        # call info
        if data_list and "call_num" in data_list[0]:
            _data = [d for d in data_list if "call_num" in d]
            call_num = sum([d["call_num"] for d in _data]) / len(_data)
            time_cost = sum([d["time_cost"] for d in _data]) / len(_data)
            line = line + f"{call_num:>5.2f}\t{time_cost:>5.2f}"

        print(line)
    print()
    return line


# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    """
    copy from DeCAF:
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def cal_exact_match_score(golden_answers: List[str], predictions: List[str]):
    if len(golden_answers) == 0:
        if len(predictions) == 0:
            return 1
        return 0
    for prediction in predictions:
        assert isinstance(prediction, str)
        em_score = ems(prediction, golden_answers)
        if em_score:
            return em_score
    return 0


def evaluation_cwq(
    save_dir,
    full=False,
    force_full=False,
    use_sample=False,
    dataset_base_dir=None,
):
    """
    metrics: exact_match
    setting: for ablation study
    """
    # small data
    small_data_ids = []
    if use_sample:
        small_data_ids = load_sample_ids("cwq", dataset_base_dir)
    elif not full:
        paths = glob(get_dataset_split_glob("cwq", dataset_base_dir=dataset_base_dir))
        for p in paths:
            _sdata = read_json(p)
            small_data_ids.extend([d["id"] for d in _sdata])
        small_data_ids = set(small_data_ids)

    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        d = read_json(p)
        if small_data_ids and d["id"] not in small_data_ids:
            continue
        d["prediction"] = extract_dialog_last_valid_observation(d)
        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if d["compositionality_type"] == "conjunction"]
    _data2 = [d for d in data if d["compositionality_type"] == "composition"]
    _data3 = [d for d in data if d["compositionality_type"] == "comparative"]
    _data4 = [d for d in data if d["compositionality_type"] == "superlative"]

    if force_full:
        bad_item = {
            "history": [{"role": "user", "content": "None"}],
            "answers": ["_None"],
            "prediction": ["None"],
        }
        if len(_data1) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data1)} empty items to conjunction"
            )
            _data1 += [bad_item] * (150 - len(_data1))
        if len(_data2) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data2)} empty items to composition"
            )
            _data2 += [bad_item] * (150 - len(_data2))
        if len(_data3) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data3)} empty items to comparative"
            )
            _data3 += [bad_item] * (150 - len(_data3))
        if len(_data4) < 150:
            logger.warning(
                f"Full data Mode! Add {150 - len(_data4)} empty items to superlative"
            )
            _data4 += [bad_item] * (150 - len(_data4))
        data = _data1 + _data2 + _data3 + _data4

    tp_data = [
        ("conjunction", _data1),
        ("composition", _data2),
        ("comparative", _data3),
        ("superlative", _data4),
        ("total      ", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)

        # metrics: exact match, As long as any one of the golden answers is in the predicted answers,
        # it will be considered correct.
        # cwq paper code adds all aliases to golden answers to calculate "exact match"
        golden_answers_add_alias = []
        for d in data:
            aliases = d["answers"] if isinstance(d["answers"], list) else [d["answers"]]
            # for a in d["answers_raw"]:
            #     aliases.append(a["answer"])
            #     aliases.extend(a["aliases"])
            aliases = sorted(set(aliases))
            golden_answers_add_alias.append(aliases)

        assert len(golden_answers_add_alias) == len(predictions)

        exact_match = (
            sum(
                [
                    cal_exact_match_score(gs, ps)
                    for gs, ps in zip(golden_answers_add_alias, predictions)
                ]
            )
            / len(predictions)
            if len(predictions) > 0
            else 0
        )
        exact_match = round(exact_match * 100, 2)

        # print q types as header, acc as value
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\texact_match"
            if data and "history" in data[0]:
                # head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
                head = head + "\tcall_num\ttime_cost"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t[{average_f1}]\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{exact_match}"

        # turn info
        # if data and "history" in data[0]:
        #     ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
        #     line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        # call info
        if data and "call_num" in data[0]:
            _data = [d for d in data if "call_num" in d]
            call_num = sum([d["call_num"] for d in _data]) / len(_data)
            time_cost = sum([d["time_cost"] for d in _data]) / len(_data)
            line = line + f"\t{call_num:.2f}\t{time_cost:.2f}"

        print(line)
    print()
    return line


# --- kqa


def _as_kqapro_verify_bool(value) -> str | None:
    """Normalize KQA Pro Verify answers: yes/no and True/False are equivalent."""
    if value is True or value == "True" or value == "yes":
        return "True"
    if value is False or value == "False" or value == "no":
        return "False"
    return None


def whether_equal(answer: str, pred: str):
    """
    check whether the two arguments are equal as attribute value
    """
    if isinstance(answer, list):
        answer = answer[0] if answer else "_None"
    if isinstance(pred, list):
        pred = pred[0] if pred else "_None"

    answer_verify = _as_kqapro_verify_bool(answer)
    pred_verify = _as_kqapro_verify_bool(pred)
    if answer_verify is not None and pred_verify is not None:
        return answer_verify == pred_verify

    pred = str(pred).replace("_", " ")
    answer = str(answer).replace("_", " ")

    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = "{} {}".format(str(v), " ".join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split("-")
            y_split = y.split("-")
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def evaluation_kqapro(save_dir: str, dataset_base_dir: str | None = None):
    """Compute per-type and overall accuracy on the fixed KQA Pro test set."""
    test_ids = load_kqapro_test_ids(dataset_base_dir)

    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for path in paths:
        d = read_json(path)
        if str(d["id"]) not in test_ids:
            continue
        if d.get("history"):
            d["prediction"] = extract_dialog_last_valid_observation_kqa(d)
        data.append(d)
    if not data:
        raise ValueError(f"No KQA Pro test predictions in {save_dir}")

    # [("type", data), ...]
    # Count   QueryAttr   QueryAttrQualifier  QueryName   QueryRelation   QueryRelationQualifier  SelectAmong     SelectBetween    Verify
    for d in data:
        d["answers"] = d.pop("answer") if "answer" in d else d["answers"]
        d["answers"] = (
            [d["answers"]] if isinstance(d["answers"], bool) else d["answers"]
        )

    _data1 = [d for d in data if d["qtype"] == "Count"]
    _data2 = [d for d in data if d["qtype"] == "QueryAttr"]
    _data3 = [d for d in data if d["qtype"] == "QueryAttrQualifier"]
    _data4 = [d for d in data if d["qtype"] == "QueryName"]
    _data5 = [d for d in data if d["qtype"] == "QueryRelation"]
    _data6 = [d for d in data if d["qtype"] == "QueryRelationQualifier"]
    _data7 = [d for d in data if d["qtype"] == "SelectAmong"]
    _data8 = [d for d in data if d["qtype"] == "SelectBetween"]
    _data9 = [d for d in data if d["qtype"] == "Verify"]

    tp_data = [
        ("Count", _data1),
        ("QueryAttr", _data2),
        ("QueryAttrQualifier", _data3),
        ("QueryName", _data4),
        ("QueryRelation", _data5),
        ("QueryRelationQualifier", _data6),
        ("SelectAmong", _data7),
        ("SelectBetween", _data8),
        ("Verify", _data9),
        ("total", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        if tp == "total":
            tp = colorful(tp, "yellow")
        print(f"{tp}: {len(data)}", end="\t")
    print()

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # follow the kqapro paper.
        # whether_equal for zip of golden_answers and predictions
        acc = (
            sum(
                [
                    whether_equal(goldn, pred)
                    for goldn, pred in zip(golden_answers, predictions)
                ]
            )
            / len(golden_answers)
            if len(golden_answers) > 0
            else 0
        ) * 100

        # print q types as header, acc as value
        if idx == 0:
            head = "\t".join([i[0] for i in tp_data])
            # turn
            if data and "history" in data[0]:
                # head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
                head = head + "\tcall_num\ttime_cost"
            print(colorful(head, "red"))
            line = [int(acc)]
            continue
        if tp == "total":
            tp = colorful(tp, "yellow")
            line.append(f"[{acc:.2f}]")
        else:
            line.append(int(acc))
        if idx == len(tp_data) - 1:
            line = "\t".join([str(i) for i in line])
            # turn stats
            # if data and "history" in data[0]:
            #     ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
            #     line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"
            if "call_num" in data[0]:
                _data = [d for d in data if "call_num" in d]
                ave_call_num = sum([d["call_num"] for d in _data]) / len(_data)
                ave_time_cost = sum([d["time_cost"] for d in _data]) / len(_data)
                line = line + f"\t{ave_call_num:.2f}\t{ave_time_cost:.2f}"
            print(line)

        # if print success_rate ave_turn_done by Qtype, comment continue; uncomment the following
        # ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
        # print(f"{tp};{ave_turn};{success_rate};{ave_turn_done}")

    print()
    return line


# --- add
def cal_turns_avef1(dataset):
    print(f"dataset: {dataset}")
    if dataset in ["cwq", "kqapro"]:
        dataset = dataset + "-addqtype"

    p = f"save-qa-infer-dialog/{dataset}/gpt-4-1106-preview/*.json"
    paths = glob(p)

    turns_avef1 = defaultdict(list)
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d)
            if "prediction" not in d
            else d["prediction"]
        )

        if "kqapro" in dataset or "metaqa" in dataset:
            d["answers"] = d.pop("answer")
            d["answers"] = (
                [d["answers"]] if isinstance(d["answers"], bool) else d["answers"]
            )

        golden_answers = clean_xsd(d["answers"])
        predictions = clean_xsd(d["prediction"])
        precision, recall, f1, random_hit = CalculatePRF1(golden_answers, predictions)
        turns = len(d["history"])
        turns_avef1[turns].append(f1)

    # print by sorted turns, format is: turns num oneline, average f1 oneline
    for turns, f1s in sorted(turns_avef1.items()):
        print(turns, end="\t")
    print()
    for turns, f1s in sorted(turns_avef1.items()):
        print(sum(f1s) * 100 / len(f1s), end="\t")
    print()


if __name__ == "__main__":
    # f1s = []
    # for i in range(2,10):
    #     model_name = f"Merge-Llama-2-13b-hf-full-zero3-epoch10/epoch_{i}"
    #     line = evaluation_webqsp(dirname="save-qa-infer-dialog-finetuned", model_name=model_name)
    #     f1s.append(line.split("\t")[3])
    cal_turns_avef1("webqsp")
    cal_turns_avef1("cwq")
    cal_turns_avef1("kqapro")
    cal_turns_avef1("metaqa")
