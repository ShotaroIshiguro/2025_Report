#!/bin/bash

# cd "diffusion-rig-EMO/DiffusionRig_main"
pwd

#############################################
# 表情転送の元となる表情を持つターゲット人物を指定
target_people=("") 
# 表情転送先となるソース人物を指定（複数人指定可能）
people=("")
# ターゲット人物の画像枚数を指定
target_num=("") 
#############################################




######################### 以下コマンドライン形式の画像生成命令 ######################### 
for target_person in "${target_people[@]}"; do
    for x in $(seq -w 1 "${target_num}"); do
        echo ""${target_person}"_"${x}".png"
        for person in "${people[@]}"; do
            echo "人物："${person}" deca_resnet18 running..."

            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet18_targetDECA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model DECA
            done

            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
        done
            echo "--------------------------------------------------------" 
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet18_targetEMOCA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model DECA
            done

            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"         
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet18_targetEMOCA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model EMOCA
            done

            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet18_targetDECA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model EMOCA
            done

            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            echo "人物："${person}" deca_resnet50 running..."

            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet50_targetDECA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------" 
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet50_targetEMOCA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"         
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet50_targetEMOCA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainDECA_resnet50_targetDECA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/deca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"   


            echo "人物："${person}" emoca_resnet18 running..."

            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet18_targetDECA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------" 
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet18_targetEMOCA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"         
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet18_targetEMOCA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet18_targetDECA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference18.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet18/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            echo "人物："${person}" emoca_resnet50 running..."

            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet50_targetDECA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------" 
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet50_targetEMOCA_sourceDECA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model DECA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"         
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet50_targetEMOCA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model EMOCA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"    
            od="output_dir_affectnet/target_"${target_person}"_"${x}"/"${person}"/trainEMOCA_resnet50_targetDECA_sourceEMOCA/"
            for i in {1..4}; do
                buffer_folder="buffer$i"
                echo "出力フォルダ：${od}${buffer_folder}"
                python scripts/inference50.py  \
                --timestep_respacing ddim50 \
                --modes exp \
                --source jisaku_training/"${person}"_aligned \
                --target jisaku_training/"${target_person}"_aligned/"${target_person}""${x}".png \
                --model_path log_affectnet/stage2/"${person}"/emoca_resnet50/model010000.pt \
                --meanshape personalLMDB/personal_deca_"${person}".lmdb/mean_shape.pkl \
                --output_dir "${od}${buffer_folder}" \
                --target_model DECA \
                --source_model EMOCA
            done


            folders=(buffer1 buffer2 buffer3 buffer4)
            most_clear_folder=$(python evaluate_sharpness.py "${folders[@]}" "${od}")

            echo "Most clear folder: $most_clear_folder"

            mv "$most_clear_folder"/* "$od"

            for folder in "${folders[@]}"; do
                rm -rf "${od}${folder}"
                echo "Deleted ${od}${folder}"
            done
            echo "--------------------------------------------------------"   

        done
    done
done
