import time
from typing import Dict, List, Tuple

import requests
from loguru import logger

from common.common_utils import catch_keyboard_interrupt
from common.constant import LOGIC_FORM_REWARD_SERVERS
from train.lf_sft import LOGIC_FORM_INSTRUCTION

_index = 0


def compute_logic_form_reward(
    messages: List[Dict], q: str, *args, **kwargs
) -> Tuple[str, float, List[int]]:
    """
    set n=10 to sample 10 times and get the mode score and the corresponding eval reason.
    Do not support ChatGPT models.
    """
    # find the last ExecuteSPARQL
    last_action = None
    for i in range(len(messages) - 1, -1, -1):
        if "ExecuteSPARQL" in messages[i]["content"]:
            last_action = messages[i]["content"].split("ExecuteSPARQL")[1].strip()[2:-2]
            break
    if last_action is None:
        return 0
    new_messages = [
        {"role": "system", "content": LOGIC_FORM_INSTRUCTION},
        {"role": "user", "content": f"Q: {q}"},
        {"role": "assistant", "content": "SPARQL: " + last_action},
    ]

    global _index
    for _ in range(3):
        try:
            url = LOGIC_FORM_REWARD_SERVERS[(_index) % len(LOGIC_FORM_REWARD_SERVERS)]
            _index += 1

            # from HF.
            # response = requests.post(
            #     f"{url}/get_logprob",
            #     json={"messages": new_messages},
            #     headers={"Content-Type": "application/json"},
            #     timeout=10,
            # )
            # response.raise_for_status()  # Raise exception if status code is not 200
            # logprob_info = response.json()["logprob"]

            # from vllm.
            response = requests.post(
                url + "/v1/chat/completions",
                json={
                    "messages": new_messages,
                    "logprobs": True,
                    "temperature": 0.0,
                    "echo": True,
                    "max_tokens": 1,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer kbqa",
                },
                timeout=10,
            )
            response.raise_for_status()

            prompt_logprobs = response.json()["prompt_logprobs"]

            # dict: {'token': str, 'logprob': float, ...}
            logprob_info = []
            for _dict in prompt_logprobs:
                if _dict:
                    _d = list(_dict.values())[0]
                    logprob_info.append(
                        {"token": _d["decoded_token"], "logprob": _d["logprob"]}
                    )
            # ------------------------- #

            # Calculate average logprob after "SPARQL:" (tokens: SP AR QL :)
            sparql_start_idx = 0
            for i, token_info in enumerate(logprob_info):
                if (
                    token_info["token"] == ":"
                    and i > 0
                    and logprob_info[i - 1]["token"] == "QL"
                    and logprob_info[i - 2]["token"] == "AR"
                    and logprob_info[i - 3]["token"] == "SP"
                ):
                    sparql_start_idx = i + 1
                    break

            if sparql_start_idx == 0 or sparql_start_idx >= len(logprob_info):
                return 0

            total_logprob = 0
            token_count = 0
            for i in range(sparql_start_idx, len(logprob_info)):
                total_logprob += logprob_info[i]["logprob"]
                token_count += 1

            if token_count == 0:
                return 0

            avg_logprob = total_logprob / token_count
            # Convert to a reward score between 0 and 1
            # Language model log probabilities are typically negative values, often between -10 and 0
            # Higher (less negative) values indicate higher confidence
            # The transformation (avg_logprob + 10) / 10 maps typical values to 0-1 range
            # This is reasonable because:
            # 1. It preserves the relative ordering of log probabilities
            # 2. It normalizes the scale for easier comparison
            # 3. It handles the typical range of log probabilities from LLMs
            reward = min(max((avg_logprob + 10) / 10, 0), 1)
            return reward * 1.2

        except Exception as e:
            logger.error(f"Error in get_logprob: {e}")
            time.sleep(1)
    return None


@catch_keyboard_interrupt
def test_reward():

    # good:
    messages2 = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Q: what electorate does anna bligh represent?"},
        {
            "role": "assistant",
            "content": 'Thought: To find out the electorate that Anna Bligh represents, we need to identify the entity "Anna Bligh" and then find the predicate that connects a person to the electorate they represented. First, let\'s search for the node related to "Anna Bligh".\nAction: SearchNodes("Anna Bligh")',
        },
        {
            "role": "user",
            "content": "Observation: ['\"Anna Bligh\" | Description: Anna Maria Bligh is an Australian politician and the 37th Premier of Queensland from 2007 to 2012. B...', '\"Anna Bligh crop\" | No description.', '\"Andrew McNamara and Anna Bligh\" | No description.', '\"William Bligh vs William Bligh\" | No description.', '\"Bligh County\" | Description: Bligh County was one of the original Nineteen Counties in New South Wales and is now one of the 141 ...', '\"Bligh Island Marine Provincial Park\" | Description: Bligh Island Marine Provincial Park is a provincial park in British Columbia, Canada.\\nThe park encom...', '\"Bligh NSW\" | No description.', '\"Bligh Park\" | Description: Bligh Park is a suburb of Sydney, in the state of New South Wales, Australia. Bligh Park is located ...', '\"Bligh Reef\" | Description: Bligh Reef, sometimes known as Bligh Island Reef, is a reef off the coast of Bligh Island in Prince ...', '\"Bligh Roosters\" | Description: The Bligh Roosters is a former Fijian rugby union team that had a franchise area covering Tavua, Vat...']",
        },
        {
            "role": "assistant",
            "content": 'Thought: The first node with the description "Anna Maria Bligh is an Australian politician and the 37th Premier of Queensland from 2007 to 2012" is the correct node for Anna Bligh. Now, I need to find the predicate that connects Anna Bligh to the electorate she represented.\nAction: SearchGraphPatterns(\'SELECT ?e WHERE { ?e ns:type.object.name "Anna Bligh"@en . }\', semantic="representative of an electorate")',
        },
        {
            "role": "user",
            "content": 'Observation: [("Queensland state election, 2009", government.election.winner, ?e), (?e, government.politician.government_positions_held -> government.government_position_held.district_represented, "Electoral district of South Brisbane"), ("Electoral district of South Brisbane", government.political_district.representatives -> government.government_position_held.office_holder, ?e), (?e, government.politician.party -> government.political_party_tenure.party, "Australian Labor Party"), ("Australian Labor Party", government.political_party.politicians_in_this_party -> government.political_party_tenure.politician, ?e), ("Queensland state election, 2009", government.parliamentary_election.resulting_head_of_government, ?e), (?e, government.politician.government_positions_held -> government.government_position_held.jurisdiction_of_office, "Queensland"), (?e, government.politician.government_positions_held -> government.government_position_held.from, "1995-07-15"^^xsd:date), (?e, government.politician.government_positions_held -> government.government_position_held.to, "2012-03-30"^^xsd:date), (?e, government.politician.government_positions_held -> government.government_position_held.governmental_body, "Legislative Assembly of Queensland")]',
        },
        {
            "role": "assistant",
            "content": 'Thought: The pattern "government.politician.government_positions_held -> government.government_position_held.district_represented" indicates the relationship between a politician and the district or electorate they represented. The observation shows that Anna Bligh was the representative of the "Electoral district of South Brisbane". Now, I can construct the SPARQL query to confirm this information.\nAction: ExecuteSPARQL(\'SELECT DISTINCT ?x WHERE { ?e ns:type.object.name "Anna Bligh"@en . ?e ns:government.politician.government_positions_held ?c . ?c ns:government.government_position_held.district_represented ?x . }\')',
        },
        {
            "role": "user",
            "content": "Observation: ['Electoral district of South Brisbane']",
        },
    ]

    # run 10 times:
    start_time = time.time()
    score = compute_logic_form_reward(
        messages2, q="what electorate does anna bligh represent?"
    )

    end_time = time.time()
    time_cost = end_time - start_time
    # print(f"Time cost: {time_cost:.2f}s")
    print(f"Score: {score}")
    # print()
    return time_cost


if __name__ == "__main__":
    # python reward_model/logic_form.py
    times = []
    for _ in range(10):
        time_cost = test_reward()
        times.append(time_cost)
    print(f"Average time cost: {sum(times) / len(times):.2f}s")
    print(f"Min time cost: {min(times):.2f}s")
    print(f"Max time cost: {max(times):.2f}s")
