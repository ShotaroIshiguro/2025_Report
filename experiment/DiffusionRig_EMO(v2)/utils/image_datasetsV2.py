import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
import torch as th

import lmdb
import pickle
from torchvision import transforms

"""
画像データセットを読み込み、前処理を施した上で、PyTorchのDataLoaderを使用してバッチごとにデータを供給
また、画像のLMDB（Lightning Memory-Mapped Database）データベースからデータを読み込み、変換を行う


LMDB（Lightning Memory-Mapped Database）は、高速で効率的なKey-Value型のデータベースです。
LMDBの主な特徴としては、メモリマップ（memory-mapped）I/Oを利用して、データをディスクから直接メモリに
マッピングすることで、高速な読み書きを実現しています。また、データをB+木（B+ tree）構造で管理しており、
ランダムアクセスにおいても効率的にデータを取得できます。
"""

def load_data(
    *,
    data_dir,
    batch_size,
    num_workers=16,
):
    """
    分散学習に対応したデータローダーを生成します。
    ImageDatasetをラップし、MPIを利用してデータを各プロセスに分割します。
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = ImageDataset(
        path=data_dir,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    while True:
        yield from loader

def load_data_local(
    *,
    data_dir,
    batch_size,
):
    """
    ローカル環境用のデータローダー。
    最初に全データをメモリに読み込み、その後ランダムにバッチを生成します。
    """
    env = lmdb.open(
        data_dir,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    if not env:
        raise IOError("Cannot open lmdb dataset", data_dir)

    with env.begin(write=False) as txn:
        # データベース内の全データの長さを取得
        length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    print("data: ", length)

    transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    transform = transforms.Compose(transform)

    zfill = 6

    data_image = []
    data_rendered = []
    data_normal = []
    data_albedo = []
    # --- 変更箇所：shapeとidentity用のリストを追加 ---
    data_shape = []
    data_identity = []
    # --- 変更ここまで ---

    # 画像、法線マップ、アルベド、レンダリングデータをLMDBから読み込む
    with env.begin(write=False) as txn:
        for index in range(length):

            key = f"image_{str(index).zfill(zfill)}".encode("utf-8")
            image_bytes = txn.get(key)

            key = f"normal_{str(index).zfill(zfill)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(zfill)}".encode("utf-8")
            albedo_bytes = txn.get(key)

            key = f"rendered_{str(index).zfill(zfill)}".encode("utf-8")
            rendered_bytes = txn.get(key)
            
            # --- 変更箇所：shapeとidentityのデータを読み込む ---
            key = f"shape_{str(index).zfill(zfill)}".encode("utf-8")
            shape_bytes = txn.get(key)

            key = f"identity_{str(index).zfill(zfill)}".encode("utf-8")
            identity_bytes = txn.get(key)
            # --- 変更ここまで ---

            buffer = BytesIO(image_bytes)
            image = Image.open(buffer)

            buffer = BytesIO(normal_bytes)
            normal = pickle.load(buffer)

            buffer = BytesIO(albedo_bytes)
            albedo = pickle.load(buffer)

            buffer = BytesIO(rendered_bytes)
            rendered = pickle.load(buffer)

            # --- 変更箇所：shapeとidentityのデータをデコード ---
            shape = pickle.load(BytesIO(shape_bytes)) if shape_bytes else None
            identity = pickle.load(BytesIO(identity_bytes)) if identity_bytes else None
            # --- 変更ここまで ---

            image = transform(image)

            data_image.append(image)
            data_normal.append(normal)
            data_albedo.append(albedo)
            data_rendered.append(rendered)
            # --- 変更箇所：リストにデータを追加 ---
            if shape is not None: data_shape.append(shape)
            if identity is not None: data_identity.append(identity)
            # --- 変更ここまで ---

    data_image = th.stack(data_image, 0)
    data_rendered = th.stack(data_rendered, 0)
    data_normal = th.stack(data_normal, 0)
    data_albedo = th.stack(data_albedo, 0)
    # --- 変更箇所：リストをテンソルに変換 ---
    if data_shape: data_shape = th.stack(data_shape, 0)
    if data_identity: data_identity = th.stack(data_identity, 0)
    # --- 変更ここまで ---

    while True:
        idxs = np.random.choice(length, batch_size, replace=False)
        batch = {
            "image": data_image[idxs],
            "rendered": data_rendered[idxs],
            "normal": data_normal[idxs],
            "albedo": data_albedo[idxs],
        }
        # --- 変更箇所：バッチにデータを追加 ---
        if len(data_shape) > 0: batch["shape"] = data_shape[idxs]
        if len(data_identity) > 0: batch["identity"] = data_identity[idxs]
        # --- 変更ここまで ---
        yield batch


class ImageDataset(Dataset):
    """
    PyTorchのDatasetクラス。LMDBからデータを1つずつ読み込む。
    分散学習のためにシャーディング（データ分割）に対応。
    """
    def __init__(
        self,
        path,
        shard=0,
        num_shards=1,
    ):
        super().__init__()

        self.zfill = 6
        self.path = path

        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        # 分散学習のためのインデックス分割
        self.idxs = [*range(self.length)][shard:][::num_shards]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):

        index = self.idxs[index]

        with self.env.begin(write=False) as txn:
            key = f"image_{str(index).zfill(self.zfill)}".encode("utf-8")
            image_bytes = txn.get(key)

            key = f"normal_{str(index).zfill(self.zfill)}".encode("utf-8")
            normal_bytes = txn.get(key)

            key = f"albedo_{str(index).zfill(self.zfill)}".encode("utf-8")
            albedo_bytes = txn.get(key)

            key = f"rendered_{str(index).zfill(self.zfill)}".encode("utf-8")
            rendered_bytes = txn.get(key)

            # --- 変更箇所：shapeとidentityのデータを読み込む ---
            key = f"shape_{str(index).zfill(self.zfill)}".encode("utf-8")
            shape_bytes = txn.get(key)

            key = f"identity_{str(index).zfill(self.zfill)}".encode("utf-8")
            identity_bytes = txn.get(key)
            # --- 変更ここまで ---

        buffer = BytesIO(image_bytes)
        image = Image.open(buffer)

        buffer = BytesIO(normal_bytes)
        normal = pickle.load(buffer)

        buffer = BytesIO(albedo_bytes)
        albedo = pickle.load(buffer)

        buffer = BytesIO(rendered_bytes)
        rendered = pickle.load(buffer)

        # --- 変更箇所：shapeとidentityのデータをデコード ---
        shape = pickle.load(BytesIO(shape_bytes)) if shape_bytes else None
        identity = pickle.load(BytesIO(identity_bytes)) if identity_bytes else None
        # --- 変更ここまで ---
        
        image = self.transform(image)
        
        batch = {
            "image": image,
            "normal": normal,
            "albedo": albedo,
            "rendered": rendered,
        }
        # --- 変更箇所：バッチにデータを追加 ---
        if shape is not None: batch["shape"] = shape
        if identity is not None: batch["identity"] = identity
        # --- 変更ここまで ---
        
        return batch
