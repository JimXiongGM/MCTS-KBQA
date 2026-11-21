#!/bin/bash
set -e
# Reward methods: IG-Beam / Classic MCTS / Fast MCTS / Fast MCTS-Const
# Usage: bash scripts/run_infer_with_reward.sh webqsp
export PYTHONPATH=./:$PYTHONPATH
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

dataset=${1:?usage: bash scripts/run_infer_with_reward.sh <webqsp|cwq|grailqa|kqapro>}
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

# Use the reward model that matches the inference server.
reward_model=${REWARD_MODEL:?Set REWARD_MODEL to the local reward model directory.}

# IG-Beam
python -m search.ig_beam \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --note v1 \
    --beam_width 5 \
    --actor_n 5 \
    --max_workers $workers \
    --save_dir $root/tree-with-reward/ig-beam-Actor-${actor_name}-v1/$dataset \
    --result_subdir "" \
    --reward_model "$reward_model"

# Classic MCTS
python -m search.classic_mcts \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --reward_method conditional_probability \
    --reward_model "$reward_model" \
    --note v1 \
    --max_workers $workers \
    --save_dir $root/tree-with-reward/classic-mcts-Actor-${actor_name}-v1/$dataset \
    --result_subdir "" \
    --pdf_subdir "pdf"

# Fast MCTS
python -m search.fast_mcts \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --note v1 \
    --max_workers $workers \
    --save_dir $root/tree-with-reward/fast-mcts-Actor-${actor_name}-v1/$dataset \
    --result_subdir "" \
    --pdf_subdir "pdf" \
    --reward_model "$reward_model"

# Fast MCTS (constant reward)
python -m search.fast_mcts \
    --dataset $dataset \
    --actor_model "$actor_model" \
    --note v1 \
    --use_constant_reward True \
    --max_workers $workers \
    --save_dir $root/tree/fast-mcts-constant-reward-${actor_name}-v1/$dataset \
    --result_subdir "" \
    --pdf_subdir "pdf"
