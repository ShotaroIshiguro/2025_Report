import os
import cv2
import numpy as np
import pandas as pd
import pywt
from tqdm import tqdm

def estimate_wavelet_noise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 0.0

    coeffs2 = pywt.dwt2(image, 'db1')  # Haar wavelet
    _, (cH, cV, cD) = coeffs2  # Horizontal, Vertical, Diagonal detail coefficients
    sigma_est = np.median(np.abs(cD)) / 0.6745  # Median Absolute Deviation (MAD)
    return sigma_est

base_dir = "output_dir_noise"
models = ["ddim", "ddpm", "dpm_solver"]
steps = ["50", "100", "1000"]
stages = ["010000", "020000"]
print("start")

# 保存用データ
model_step_data = {model: [] for model in models}
stage2_data = {model: [] for model in models}
meanshape_data = {model: [] for model in models}

for model in models:
    for step in steps:
        for stage in stages:
            # meanshapeなし
            key_plain = f"{model}_{step}_{stage}"
            noise_list_plain = []

            # meanshapeあり
            key_mean = f"{model}_{step}_{stage}_meanshape"
            noise_list_mean = []

            for i in range(1, 51):
                index = f"{i:02d}"

                # meanshapeなし画像
                img_dir_plain = os.path.join(base_dir, key_plain, index)
                # meanshapeあり画像
                img_dir_mean = os.path.join(base_dir, key_mean, index)

                for dir_path, lst in [(img_dir_plain, noise_list_plain), (img_dir_mean, noise_list_mean)]:
                    if not os.path.exists(dir_path):
                        continue

                    img_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    if img_files:
                        image_path = os.path.join(dir_path, img_files[0])
                        lst.append(estimate_wavelet_noise(image_path))

            # 各評価値（平均）
            noise_mean_plain = np.mean(noise_list_plain) if noise_list_plain else 0
            noise_mean_mean = np.mean(noise_list_mean) if noise_list_mean else 0

            model_step_data[model].append(noise_mean_plain)

            if step == "1000":
                stage2_data[model].append(noise_mean_plain)

            if step == "1000" and stage == "010000":
                meanshape_data[model].append(noise_mean_plain)
                meanshape_data[model].append(noise_mean_mean)

# ==== CSV出力 ====

# 1. model_step.csv: 行=Steps, 列=models
model_step_df = pd.DataFrame(model_step_data, index=steps * len(stages))
model_step_df.to_csv("model_step.csv")

# 2. stage2.csv: 行=stages, 列=models（step=1000固定）
stage2_df = pd.DataFrame(stage2_data, index=stages)
stage2_df.to_csv("stage2.csv")

# 3. meanshape.csv: 行=["no_meanshape", "meanshape"], 列=models（step=1000, stage=010000固定）
meanshape_df = pd.DataFrame(meanshape_data, index=["no_meanshape", "meanshape"])
meanshape_df.to_csv("meanshape.csv")
