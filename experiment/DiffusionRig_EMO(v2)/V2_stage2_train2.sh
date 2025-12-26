#!/bin/bash


people=("hitoshi" "guro" "hayata" "obama" "trump" "ohtani" "shiraishi" "hamabe" "hamada" "haris" "hashikan" "gaki" "yonedu" "dewi" "zuck" "altman" "setokan" "jensen" "merkel" "pichai" "elen" "elon")
models=("deca" "emoca")
for person in "${people[@]}"; do
    for model in "${models[@]}"; do
        # resnet 18or50 & source DECAorEMOCA
        echo "人物："${person}" "

        python scripts/trainV2.py \
            --log_dir logV2/stage2/"${person}"/"${model}"_resnet18_V2 \
            --resume_checkpoint logV2/stage1/"${model}"_affectnet_resnet18_V2/model100000.pt \
            --data_dir personalLMDB_V2/"${model}"_"${person}"_V2.lmdb \
            --lr 1e-5 \
            --p2_weight True \
            --image_size 256 \
            --batch_size 4 \
            --max_steps 15000 \
            --num_workers 6 \
            --save_interval 5000 \
            --stage 2 \
            --encoder_type resnet18 \
            --latent_dim 64 \
            --shape_dim 100 \
            --identity_dim 150

        python scripts/trainV2.py \
            --log_dir logV2/stage2/"${person}"/"${model}"_resnet50_V2 \
            --resume_checkpoint logV2/stage1/"${model}"_affectnet_resnet50_V2/model100000.pt \
            --data_dir personalLMDB_V2/"${model}"_"${person}"_V2.lmdb \
            --lr 1e-5 \
            --p2_weight True \
            --image_size 256 \
            --batch_size 4 \
            --max_steps 15000 \
            --num_workers 6 \
            --save_interval 5000 \
            --stage 2 \
            --encoder_type resnet50 \
            --latent_dim 64 \
            --shape_dim 100 \
            --identity_dim 150
    done
done

