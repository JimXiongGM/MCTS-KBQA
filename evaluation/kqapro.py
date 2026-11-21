import json
from glob import glob
from typing import List

from common.path_utils import load_kqapro_test_ids


def extract_answer(d):
    if not d:
        return None
    if isinstance(d["history"][-1], str):
        last_content = d["history"][-1]
    elif isinstance(d["history"][-1], dict):
        last_content = d["history"][-1]["content"]
    else:
        raise ValueError

    pred_ans = None
    if last_content == "Stop condition detected.":
        # use the last Observation
        try:
            last_obs = (
                d["history"][-3]["content"]
                .replace("Observation: ", "")
                .replace("\nThought:", "")
            )
            pred_ans = eval(last_obs)

            # ['2024-01-01"^^xsd:date'] -> ['2024-01-01']
            if isinstance(pred_ans, list):
                for i in range(len(pred_ans)):
                    if "^^xsd" in pred_ans[i]:
                        pred_ans[i] = pred_ans[i].split("^^xsd")[0].strip('"')
                pred_ans = sorted(set(pred_ans))
            elif isinstance(pred_ans, bool):
                pass
            else:
                raise ValueError("Unknown type: ", pred_ans)
        except:
            # print("eval error: ", last_obs)
            pass
        if pred_ans:
            return pred_ans
    return ["_None"]


def cal_acc(data: List):
    accuracys = []
    for d in data:
        prediction = str(extract_answer(d))
        golden = str(d["answer"])
        good = 1 if prediction == golden else 0
        accuracys.append(good)
    ave_acc = sum(accuracys) / len(accuracys)
    if accuracys:
        print("total items:", len(accuracys), end=" ")
        print("acc:", round(ave_acc, 4))
    return ave_acc


def main():
    """Compute accuracy on the fixed KQA Pro test questions."""
    import os

    # p = "save-qa/kqapro/Mistral-7B-v0.1-epoch10-v1.0"
    # p = "save-qa/kqapro/Mistral-7B-v0.1-epoch10-v2.0-n30"
    # p = "save-qa/kqapro/Llama-2-7b-hf-epoch10-v1.0"
    p = "save-qa/kqapro/Llama-2-7b-hf-epoch10-v2.0-n30"

    # excel format.
    # name, Count, QueryAttrQualifier, QueryAttr, QueryName, QueryRelationQualifier, QueryRelation, SelectAmong, SelectBetween, Verify, total, Average
    head = "name\tCount\tQueryAttrQualifier\tQueryAttr\tQueryName\tQueryRelationQualifier\tQueryRelation\tSelectAmong\tSelectBetween\tVerify\ttotal\tAverage"
    name = os.path.basename(p)
    outs = [name]

    print(p)
    paths = glob(p + "/*.json")
    test_ids = load_kqapro_test_ids()
    data = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            item = json.load(file)
        if str(item["id"]) in test_ids:
            data.append(item)

    qtypes = [
        "Count",
        "QueryAttrQualifier",
        "QueryAttr",
        "QueryName",
        "QueryRelationQualifier",
        "QueryRelation",
        "SelectAmong",
        "SelectBetween",
        "Verify",
    ]
    for qtype in qtypes:
        _data = [d for d in data if d["qtype"] == qtype]
        print(f"qtype: {qtype}", end=" ")
        ave_acc = cal_acc(_data)
        outs.append(ave_acc)

    outs.append(len(data))
    print("Average accuracy:", end=" ")
    ave_acc = cal_acc(data)
    outs.append(ave_acc)

    print(head)
    print("\t".join([str(x) for x in outs]))


if __name__ == "__main__":
    main()
