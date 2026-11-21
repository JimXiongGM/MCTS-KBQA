#!/bin/bash
set -e
# No-reward methods: linear (5 seeds) + BFS/DFS/RandomWalk + major-vote
# Usage: bash scripts/run_infer_no_reward.sh webqsp
export PYTHONPATH=./:$PYTHONPATH

dataset=${1:?usage: bash scripts/run_infer_no_reward.sh <webqsp|cwq|grailqa|kqapro>}
root=save-fast-mcts-kbqa
workers=4

if [[ $dataset == grailqa ]]; then
    actor_model=save-model-sft/small-GrailQA-v2/Meta-Llama-3.1-8B-Instruct-epoch5
elif [[ $dataset == kqapro ]]; then
    actor_model=save-model-sft/small-KQAPro-v2/Meta-Llama-3.1-8B-Instruct-epoch5
elif [[ $dataset == webqsp || $dataset == cwq ]]; then
    actor_model=save-model-sft/small-WebQSP+CWQ-v2/Meta-Llama-3.1-8B-Instruct-epoch5
else
    echo "dataset must be webqsp, cwq, grailqa, or kqapro"
    exit 1
fi
actor_model=${ACTOR_MODEL:-$actor_model}
actor_name=$(basename "$actor_model")

# linear (5 seeds)
for seed in 1 10 100 1000 10000; do
    python -m search.linear \
        --dataset $dataset \
        --model_name "$actor_model" \
        --seed $seed \
        --max_workers $workers \
        --save_dir $root/linear/${actor_name}-seed${seed}/$dataset
done

# linear major-vote
python -m evaluation.linear_vote \
    --path $root/linear/${actor_name}-seed1 \
    --dataset $dataset \
    --output_path $root/linear/${actor_name}-major-vote/$dataset

# BFS
python -m search.tree \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --select_method bfs \
    --note v1 \
    --max_workers $workers \
    --save_dir $root/tree/bfs-${actor_name}-v1/$dataset \
    --result_subdir ""

# DFS
python -m search.tree \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --select_method dfs \
    --note v1 \
    --max_workers $workers \
    --save_dir $root/tree/dfs-${actor_name}-v1/$dataset \
    --result_subdir ""

# RandomWalk
python -m search.tree \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --select_method randomwalk \
    --note v1 \
    --max_workers $workers \
    --save_dir $root/tree/randomwalk-${actor_name}-v1/$dataset \
    --result_subdir ""
