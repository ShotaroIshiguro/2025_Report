#!/bin/bash

cd "diffusion-rig-EMO/DiffusionRig_main"

#############################################
# 表情転送先となるソース人物を指定（複数人指定可能）
people=("")
#############################################




######################### 以下コマンドライン形式の学習実行命令 ######################### 
for person in "${people[@]}"; do
    # resnet 18or50 & source DECAordeca
    echo "人物："${person}" "
    # mpiexec -n 2 
    python scripts/train18.py --latent_dim 64 --encoder_type resnet18 \
        --log_dir log_affectnet/stage2/"${person}"/deca_resnet18 --data_dir personalLMDB/personal_deca_"${person}".lmdb --lr 1e-5 \
        --p2_weight True --image_size 256 --batch_size 4  --max_steps 20000 \
        --num_workers 8 --save_interval 5000 --stage 2 \
        --resume_checkpoint log_affectnet/stage1/deca_affectnet_resnet18_batch8/model100000.pt

    # mpiexec -n 2 
    python scripts/train50.py --latent_dim 64 --encoder_type resnet50 \
        --log_dir log_affectnet/stage2/"${person}"/deca_resnet50 --data_dir personalLMDB/personal_deca_"${person}".lmdb --lr 1e-5 \
        --p2_weight True --image_size 256 --batch_size 4  --max_steps 20000 \
        --num_workers 8 --save_interval 5000 --stage 2 \
        --resume_checkpoint log_affectnet/stage1/deca_affectnet_resnet50_batch8/model100000.pt
done


