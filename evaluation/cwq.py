import json
from glob import glob

from evaluation.runtime import extract_dialog_last_valid_observation

extract_LLM_answer = extract_dialog_last_valid_observation


def extract_golden_answers(d):
    ans = [a["answer"] for a in d["answers"]]
    return ans


def main(num=None):
    """
    python evaluation/kqapro.py --num=100
    """
    _paths = ["save/cwq/4types-gpt-3.5-turbo-16k-v1.0"]
    for _i in range(len(_paths)):
        paths = glob(_paths[_i] + "/train/*.jsonl")[:]
        accu_good, total = 0, 0
        for path in paths:
            # debug
            # if "QueryName" not in path:
            #     continue

            print(path.split("cwq/")[-1], end="  ")
            data = [json.loads(i) for i in open(path).readlines()]
            data = [d for d in data if d]
            predictions = [
                extract_dialog_last_valid_observation(d) if d else ["None"]
                for d in data[:num]
            ]
            goldens = [extract_golden_answers(d) for d in data[:num]]
            assert len(predictions) == len(goldens)
            good = sum(
                [1 if set(p) == set(g) else 0 for p, g in zip(predictions, goldens)]
            )
            print("total:", len(goldens), end="  ")
            print("acc:", round(good / len(goldens), 4))
            accu_good += good
            total += len(goldens)
        print("total items:", total)
        print("total acc:", round(accu_good / total, 4))
        print()


if __name__ == "__main__":
    main()
    # fire.Fire(main)
