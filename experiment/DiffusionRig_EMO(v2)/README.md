# DiffusionRig-Emo(v2)

DiffusionRig-Emo(v1)の顔の物理的条件を規定するDECAをEMOCAに置き換えることで、より現実的な表情変換を可能にしたプロジェクトです。

*Shotaro Ishiguro*

![Image 1](https://github.com/ShotaroIshiguro/2025Report/blob/main/architecture.png)

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
FFHQとAffectNetの画像から3D顔特徴を事前に抽出します。
```
cd DiffusionRig_main
# FFHQ(DECA)
python scripts/create_data.py --data_dir FFHQ/FFHQ_images \
    --output_dir ffhq256_deca.lmdb --image_size 256 --use_meanshape False \
    --use_model DECA
# FFHQ(EMOCA)
python scripts/create_data.py --data_dir FFHQ/FFHQ_images \
    --output_dir ffhq256_emoca.lmdb --image_size 256 --use_meanshape False \
    --use_model EMOCA
```

### Stage2のデータセット
個人のアルバムに対して、顔の整列（Alignment）と物理バッファの抽出を行います。
```
conda deactivate DRE
conda env create -n DRE_align python=3.9 --file environment_DRE_align.yml
conda activate DRE_align

python scripts/align.py -i PATH_TO_PERSONAL_PHOTO_ALBUM \
    -o PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM -s 256

conda deactivate DER_align
conda activate DRE

# Personal_Album(DECA)
python scripts/create_data.py --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 --use_meanshape True --use_model DECA
# Personal_Album(EMOCA)
python scripts/create_data.py --data_dir PATH_TO_PERSONAL_ALIGNED_PHOTO_ALBUM \
    --output_dir NAME_MODEL.lmdb --image_size 256 --use_meanshape True --use_model EMOCA
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
```
python scripts/train.py --latent_dim 64 --encoder_type resnet18  \
    --log_dir log/stage1/emoca_FFHQ_resnet18_batch16 --data_dir ffhq256_emoca.lmdb \
    --lr 1e-4 --p2_weight True --image_size 256 --batch_size 16 --max_steps 50000 \
    --num_workers 8 --save_interval 5000 --stage 1
```

### ステージ2：パーソナライズされた事前学習 (Learning Personalized Priors)
少数の個人用アルバム画像を用いて、特定の人物の容姿を詳細に再現するためのファインチューニングを行います。
```
python scripts/train.py --latent_dim 64 --encoder_type resnet18 \
    --log_dir log/stage2 --resume_checkpoint log/stage1/[MODEL_NAME].pt \
    --data_dir NAME_MODEL.lmdb --lr 1e-5 \
    --p2_weight True --image_size 256 --batch_size 4 --max_steps 5000 \
    --num_workers 8 --save_interval 5000 --stage 2
```

## Inference
物理バッファ（Physical Buffer）に基づき、表情 (Exp)、ポーズ (Pose)、照明 (Light) の3要素を編集できます。
- 特徴の転送: ターゲット画像から特定の要素（顔の向き、表情、光の当たり方）のみを抽出し、ソース画像に適用します。
- ソース画像の制限: ソース画像は必ずステージ2の学習で使用した個人アルバムから選択してください。
- 複数処理: `--source` にディレクトリを指定することで、複数のソース画像をまとめて処理できます。
- モデルの混在: ターゲット画像とソース画像の物理バッファを、それぞれ `DECA` または `EMOCA` のどちらで取得するか個別に指定可能です。
- ResNet設定に関する注意点:
    - 推論コードには、`ResNet18`か`ResNet50`かを自動で切り替える機能がまだ実装されていません。
    - ResNet18で学習したモデルを使用する場合は、`utils/script_util.py` の `def model_and_diffusion_defaults()` 内にある、56行目と57行目の変数 を使用するResNetの種類に合わせて手動で書き換える必要があります。

```
python scripts/inference.py --source jisaku_training/Hitoshi_aligned/ \
   --modes exp --model_path log/stage2/stage2_model005000_Hitoshi.pt \
   --timestep_respacing ddim20 \
   --meanshape personal_deca_Hitoshi.lmdb/mean_shape.pkl \
   --target jisaku_training/obama_aligned/obama_12.png \
   --output_dir output_dir/target_smile_OBAMA12/targetEMOCA_sourseDECA \
   --target_model EMOCA \
   --source_model DECA
```

## EMOCAを用いた3D顔形状の取得
`gdl_apps/EMOCA/demos/test_emoca_on_images.py` を実行することで、2D画像から以下のデータを取得できます。
- 3D顔モデル（Mesh）
- 潜在コード（Latent Code）
- レンダリング結果

```
python demos/test_emoca_on_images.py --input_folder demos/test_images \
    --output_folder demos/output --model_name EMOCA_v2_lr_mse_20 \
    --save_mesh True --save_codes True
```

## Training以降をまとめて実行したい場合(AffectNetのみ対応)
### stage1の学習
- DECA or EMOCA, ResNet18 or ResNet50の4パターンに対応  
```./stage1_scripts_run.sh```
### stage2の学習
- 個人アルバムを作成してから実行
- `people`変数に人物名を入れる  
```./stage2_run_all.sh```
### 推論
- 1枚のターゲット画像に対して`ソース画像人物数×16(モデル数)×ソース人物パターン数`の画像が生成
```./stage3_shapnessinference.sh```


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

