#!/bin/bash
set -euo pipefail

# Let each DeepSpeed process import project modules when launched from the repository root.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

model=${1:?Usage: bash scripts/run_sft.sh /path/to/base-model [webqsp|cwq|grailqa|kqapro|all]}
dataset=${2:-all}
epoch=5
model_name=$(basename -- "$model")
sft_root=${SFT_ROOT:-save-model-sft}
if [[ ! -d $model ]]; then
    echo "Local base model directory does not exist: $model" >&2
    exit 1
fi

# Train all three actors with the same configuration. WebQSP and CWQ share an actor.
run_sft() {
    local group=$1
    shift
    deepspeed train/sft_main.py --ds_config_file config/adamw.json \
        --model_name_or_path "$model" \
        --dataset_patterns "$@" \
        --learning_rate 1e-5 \
        --warmup_ratio 0 \
        --mode interactive \
        --num_train_epochs "$epoch" \
        --gradient_checkpointing \
        --output_dir "$sft_root/$group/$model_name"
}

case "$dataset" in
    all|webqsp|cwq|grailqa|kqapro) ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
esac
if [[ $dataset == all || $dataset == webqsp || $dataset == cwq ]]; then
    run_sft small-WebQSP+CWQ-v2 "data/human-anno-v2/webqsp/*/*.json" "data/human-anno-v2/cwq/*/*.json"
fi
if [[ $dataset == all || $dataset == grailqa ]]; then
    run_sft small-GrailQA-v2 "data/human-anno-v2/grailqa/*/*.json"
fi
if [[ $dataset == all || $dataset == kqapro ]]; then
    run_sft small-KQAPro-v2 "data/human-anno-v2/kqapro/*/*.json"
fi
