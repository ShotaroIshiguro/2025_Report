#!/bin/bash


people=("guro" "hayata" "obama" "trump" "ohtani" "shiraishi" "hamabe" "hamada" "haris" "hashikan" "gaki" "yonedu" "dewi" "zuck" "altman" "setokan" "jensen" "merkel" "pichai" "elen" "elon")

for person in "${people[@]}"; do
    echo "人物："${person}" "

    python scripts/create_dataV2.py \
        --data_dir jisaku_training/"${person}"_aligned \
        --output_dir personalLMDB_V2/deca_"${person}"_V2.lmdb \
        --image_size 256 \
        --use_meanshape True \
        --use_model DECA

    python scripts/create_dataV2.py \
        --data_dir jisaku_training/"${person}"_aligned \
        --output_dir personalLMDB_V2/emoca_"${person}"_V2.lmdb \
        --image_size 256 \
        --use_meanshape True \
        --use_model EMOCA

done


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

people=("guro" "hayata" "obama" "trump" "ohtani" "shiraishi" "haris")
train_models=("deca" "emoca")
target_models=("deca" "emoca")
source_models=("deca" "emoca")

for x in $(seq -w 1 20); do
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