# DiffusionRig-Emo(v2)

DiffusionRig-Emo(v1)に、DECA及びEMOCAのパラメーターを拡散モデルに直接条件付けする機構を追加することで、画像生成開始時に失われる特徴量を減少させたモデル。

*Shotaro Ishiguro*

![Image 1](https://github.com/ShotaroIshiguro/2025_Report/blob/ede7c8b61d47686f11ea833d1f04971ea55a9ea4/experiment/DiffusionRig_EMO(v1)/architecture01.png)

![Image 1](https://github.com/ShotaroIshiguro/2025_Report/blob/7b0a60931fcb480904d728d3f6bc4edb859a34b5/experiment/DiffusionRig_EMO(v1)/architecture02.png)

## セットアップと準備

まずは環境構築を行います。
```
conda env create -n DRE python=3.8 --file environment_DRE.yml
conda activate DRE
cd DiffusionRig_main
pip install -e .
```

omegaconfでエラーが発生した場合： 以下のコマンドを試してください。
```
pip install "pip<24.1"
pip install omegaconf==2.0.5 hydra-core==1.0.7
```

Cythonが正しくインストールされない場合： 個別にインストールします。
```
pip install Cython==0.29.14
```

pytorch3dのインストールに失敗する場合：　個別にインストールします。
```
conda install pytorch3d -c pytorch3d
```

## データ準備
学習には FFHQ と AffectNet（第1ステージ）、および個人の写真アルバム（第2ステージ）を使用します。学習前に、DECAまたはEMOCAを使用して物理バッファを抽出する必要があります。

### DECAのセットアップ
学習用のデータ準備の前に、DECAのソースファイルとチェックポイントをダウンロードして設定します。 ※FLAMEリソースのダウンロードにはアカウント作成が必要です。

1. `deca_model.tar`:　[DECAのリポジトリ](https://github.com/yfeng95/DECA) から事前学習済みモデルをダウンロード。
2. `generic_model.pkl`: [FLAME公式サイト](https://flame.is.tue.mpg.de/download.php) から「FLAME 2020」をダウンロードし抽出。
3. `FLAME_texture.npz`: [FLAME公式サイト](https://flame.is.tue.mpg.de/login.php) から「FLAME texture space」をダウンロードし抽出。
4. その他、[DECAのデータページ](https://github.com/yfeng95/DECA/tree/master/data) にある以下のファイルを data/ フォルダに配置:

```
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```

### EMOCAのセットアップ
3D形状作成のためのアセットをダウンロードします。実行すると assets ディレクトリが作成されます。
```
cd gdl_apps/EMOCA/demos
bash download_assets.sh
```

### Stage1のデータセット
AffectNetの画像から3D顔特徴を事前に抽出します。
```
cd DiffusionRig_main
# DECA
python scripts/create_dataV2.py \
    --data_dir AffectNet/train_set/images \
    --output_dir affectnet256_deca.lmdb --image_size 256 \
    --use_meanshape False \
    --use_model DECA
# EMOCA
python scripts/create_dataV2.py \
    --data_dir AffectNet/train_set/images \
    --output_dir affectnet256_emoca.lmdb --image_size 256 \
    --use_meanshape False \
    --use_model EMOCA
```

### Stage2のデータセット
個人のアルバムに対して、顔の整列（Alignment）と物理バッファの抽出を行います。
```
conda deactivate DRE
conda env create -n DRE_align python=3.9 --file environment_DRE_align.yml
conda activate DRE_align

python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM \
    -o PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    -s 256

conda deactivate DER_align
conda activate DRE

# Personal_Album(DECA)
python scripts/create_dataV2.py \
    --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 \
    --use_meanshape True \
    --use_model DECA
# Personal_Album(EMOCA)
python scripts/create_dataV2.py \
    --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 \
    --use_meanshape True \
    --use_model EMOCA
```

## Training
このプロセスは、まず一般的な顔の特徴を学び、次に特定の個人の特徴を学習させる2段階（2ステージ）構成になっています。

### ステージ1：汎用的な顔の事前学習 (Learning Generic Face Priors)
不特定多数の顔に共通する特徴を学習するフェーズです。
- エンコーダーの選択: グローバルエンコーダーとして`resnet18` または `resnet50` を選択できます。
- 潜在空間の次元: `latent_dim` は拡散モデル（Diffusion Model）における潜在変数の次元数を指定します。
- 効率的なデータ管理: レンダリングされた画像や物理バッファ（形状データなど）を高速に処理するため、LMDB 形式のファイルを使用します。
- 分散学習: mpiexec を利用した複数GPUなどによる分散学習が可能です。
- 学習の再開: 途中で止まった学習を再開したい場合は、引数に `--resume_checkpoint` を追加してください。
- `shape_dim`: 拡散モデルに直接注入する形状パラメータの次元数
- `identity_dim`: 拡散モデルに直接注入するアイデンティティパラメータの次元数
```
python scripts/trainV2.py \
    --latent_dim 64 \
    --encoder_type RESNET_18_OR_50 \
    --log_dir log/stage1/[PATH_TO_STAGE1_LOG] \
    --data_dir PATH_TO_STAGE1_DATASET_LMDB \
    --lr 1e-4 \
    --p2_weight True \
    --image_size 256 \
    --batch_size 16 \
    --max_steps 50000 \
    --num_workers 8 \
    --save_interval 5000 \
    --shape_dim 100 \
    --identity_dim 150 \
    --stage 1
```

### ステージ2：パーソナライズされた事前学習 (Learning Personalized Priors)
少数の個人用アルバム画像を用いて、特定の人物の容姿を詳細に再現するためのファインチューニングを行います。
- `shape_dim`: 拡散モデルに直接注入する形状パラメータの次元数
- `identity_dim`: 拡散モデルに直接注入するアイデンティティパラメータの次元数
```
python scripts/trainV2.py \
    --latent_dim 64 \
    --encoder_type RESNET_18_OR_50 \
    --log_dir log/stage2 \
    --resume_checkpoint log/stage1/[MODEL_NAME].pt \
    --data_dir NAME_MODEL.lmdb \
    --lr 1e-5 \
    --p2_weight True \
    --image_size 256 \
    --batch_size 4 \
    --max_steps 5000 \
    --num_workers 8 \
    --save_interval 5000 \
    --shape_dim 100 \
    --identity_dim 150 \
    --stage 2
```

## Inference
物理バッファ（Physical Buffer）に基づき、表情 (Exp)、ポーズ (Pose)、照明 (Light) の3要素を編集できます。
- 特徴の転送: ターゲット画像から特定の要素（顔の向き、表情、光の当たり方）のみを抽出し、ソース画像に適用します。
- ソース画像の制限: ソース画像は必ずステージ2の学習で使用した個人アルバムから選択してください。
- 複数処理: `--source` にディレクトリを指定することで、複数のソース画像をまとめて処理できます。
- モデルの混在: ターゲット画像とソース画像の物理バッファを、それぞれ `DECA` または `EMOCA` のどちらで取得するか個別に指定可能です。

```
python scripts/inference_[18_OR_50]_V2.py \
    --source PATH_TO_SOURCE_IMAGES \
    --modes exp \
    --model_path [PATH_TO_PERSONAL_MODEL].pt \
    --timestep_respacing ddpm \
    --meanshape [PATH_TO_PERSONAL_DATASET].lmdb/mean_shape.pkl \
    --target PATH_TO_TARGET_IMAGE \
    --output_dir PATH_TO_OUTPUT_DIR \
    --target_model DECA_OR_EMOCA \
    --source_model DECA_OR_EMOCA
```


## 参考文献
1. DiffusionRig: Learning Personalized Priors for Facial Appearance Editing  
CVPR 2023  
https://arxiv.org/pdf/2304.06711  
https://github.com/adobe-research/diffusion-rig
```
@misc{ding2023diffusionriglearningpersonalizedpriors,
      title={DiffusionRig: Learning Personalized Priors for Facial Appearance Editing}, 
      author={Zheng Ding and Xuaner Zhang and Zhihao Xia and Lars Jebe and Zhuowen Tu and Xiuming Zhang},
      year={2023},
      eprint={2304.06711},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2304.06711}, 
}
```
2. EMOCA: Emotion Driven Monocular Face Capture and Animation  
CVPR 2022  
https://arxiv.org/pdf/2204.11312  
https://github.com/radekd91/emoca
```
@misc{danecek2022emocaemotiondrivenmonocular,
      title={EMOCA: Emotion Driven Monocular Face Capture and Animation}, 
      author={Radek Danecek and Michael J. Black and Timo Bolkart},
      year={2022},
      eprint={2204.11312},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.11312}, 
}
```

3. AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild  
IEEE Transactions on Affective Computing, 2017  
http://mohammadmahoor.com/affectnet/  
https://github.com/djordjebatic/AffectNet  
```
@ARTICLE{8013713,
    author={A. Mollahosseini and B. Hasani and M. H. Mahoor},
    journal={IEEE Transactions on Affective Computing},
    title={AffectNet: A Database for Facial Expression, Valence, and Arousal
    Computing in the Wild},
    year={2017},
    volume={PP},
    number={99},
    pages={1-1},}
```

4. Learning an Animatable Detailed 3D Face Model from In-The-Wild Images  
SIGGRAPH 2021  
https://arxiv.org/abs/2012.04012  
https://github.com/yfeng95/DECA  
```
@misc{feng2021learninganimatabledetailed3d,
      title={Learning an Animatable Detailed 3D Face Model from In-The-Wild Images}, 
      author={Yao Feng and Haiwen Feng and Michael J. Black and Timo Bolkart},
      year={2021},
      eprint={2012.04012},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2012.04012}, 
}
```
5. A Style-Based Generator Architecture for Generative Adversarial Networks(FFHQ)  
CVPR 2019 final version  
https://arxiv.org/pdf/1812.04948  
https://github.com/NVlabs/ffhq-dataset  
```
@misc{karras2019stylebasedgeneratorarchitecturegenerative,
      title={A Style-Based Generator Architecture for Generative Adversarial Networks}, 
      author={Tero Karras and Samuli Laine and Timo Aila},
      year={2019},
      eprint={1812.04948},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1812.04948}, 
}
```

