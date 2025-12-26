
import os
import cv2
import numpy as np
import sys

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

def evaluate_folder(folder):
    sharpness_values = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            sharpness = calculate_sharpness(file_path)
            sharpness_values.append(sharpness)
    return np.mean(sharpness_values) if sharpness_values else 0

folders = sys.argv[1:-1]  # 最後の引数を除いたすべての引数がフォルダリスト
od = sys.argv[-1]  # 最後の引数がod
sharpness_results = {}

for folder in folders:
    full_path = os.path.join(od, folder)
    sharpness_results[full_path] = evaluate_folder(full_path)

best_folder = min(sharpness_results, key=sharpness_results.get)
print(best_folder)

