#!/bin/bash

people=("hitoshi")
train_models=("deca" "emoca")
target_models=("deca" "emoca")
source_models=("deca" "emoca")

for x in $(seq -w 6 20); do
    echo "obama_"${x}".png"
    for person in "${people[@]}"; do
        echo "人物："${person}" deca_resnet18 running..."
        for train in "${train_models[@]}"; do
            for target in "${target_models[@]}"; do
                for source in "${source_models[@]}"; do
                    python scripts/inference50_V2.py \
                    --source jisaku_training/"${person}"_aligned/ \
                    --target jisaku_training/obama_aligned/obama"${x}".png \
                    --output_dir output_dirV2/target_obama_"${x}"/"${person}"/train"${train}"_resnet50_target"${target}"_source"${source}" \
                    --modes exp \
                    --sampler ddpm \
                    --target_model "${target}" \
                    --source_model "${source}" \
                    --model_path logV2/stage2/"${person}"/"${train}"_resnet50_V2/model110000.pt
                
                    python scripts/inference18_V2.py \
                    --source jisaku_training/"${person}"_aligned/ \
                    --target jisaku_training/obama_aligned/obama"${x}".png \
                    --output_dir output_dirV2/target_obama_"${x}"/"${person}"/train"${train}"_resnet18_target"${target}"_source"${source}" \
                    --modes exp \
                    --sampler ddpm \
                    --target_model "${target}" \
                    --source_model "${source}" \
                    --model_path logV2/stage2/"${person}"/"${train}"_resnet18_V2/model110000.pt
                done
            done
        done
    done
done