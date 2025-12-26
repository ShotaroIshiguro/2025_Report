from gdl_apps.EMOCA.utils.load import load_model
from gdl.utils.FaceDetector import FAN
from gdl.datasets.FaceVideoDataModule import TestFaceVideoDM
import gdl
import matplotlib.pyplot as plt
import gdl.utils.DecaUtils as util
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from gdl.utils.lightning_logging import _fix_image

"""
入力画像から3D顔モデル(形状・表情・テクスチャ)を復元
    メッシュデータをOBJファイルで保存
    形状・姿勢・表情などをNumPy形式で保存
    可視化画像を保存
"""

# 画像データの処理に使用、Pytorchをnumpyへ
def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

# 3D顔メッシュをOBJで保存
def save_obj(emoca, filename, opdict, i=0):
    # 詳細メッシュの生成に使用するテンプレデータをロード
    dense_template_path = Path(gdl.__file__).parents[1] / 'assets' / "DECA" / "data" / 'texture_data_256.npy'
    dense_template = np.load(dense_template_path, allow_pickle=True, encoding='latin1').item()

    # 復元結果opdictから頂点座標・面情報・テクスチャを取り出す
    vertices = opdict['verts'][i].detach().cpu().numpy()
    faces = emoca.deca.render.faces[0].detach().cpu().numpy()
    texture = util.tensor2image(opdict['uv_texture_gt'][i])
    #　面座標、法線マップを取り出す
    uvcoords = emoca.deca.render.raw_uvcoords[0].detach().cpu().numpy()
    uvfaces = emoca.deca.render.uvfaces[0].detach().cpu().numpy()
    # save coarse mesh, with texture and normal map
    normal_map = util.tensor2image(opdict['uv_detail_normals'][i] * 0.5 + 0.5)

    # 通常のメッシュをobj形式で保存
    util.write_obj(filename, vertices, faces,
                   texture=texture,
                   uvcoords=uvcoords,
                   uvfaces=uvfaces,
                   normal_map=normal_map)
    
    # ディスプレイスメントマップと法線マップを使用した詳細メッシュを生成
    # upsample mesh, save detailed mesh
    texture = texture[:, :, [2, 1, 0]]
    normals = opdict['normals'][i].detach().cpu().numpy()
    displacement_map = opdict['displacement_map'][i].detach().cpu().numpy().squeeze()
    # gdl.utils.DecaUtilsのupsanple_mesh関数で生成
    dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture,
                                                                   dense_template)
    # 詳細メッシュの保存
    util.write_obj(filename.replace('.obj', '_detail.obj'),
                   dense_vertices,
                   dense_faces,
                   colors=dense_colors,
                   inverse_face_order=True)

# 可視化画像を指定フォルダに保存
def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs.png",  _fix_image(torch_img_to_np(vis_dict['inputs'][i])))
    imsave(final_out_folder / f"geometry_coarse.png",  _fix_image(torch_img_to_np(vis_dict['geometry_coarse'][i])))
    imsave(final_out_folder / f"geometry_detail.png", _fix_image(torch_img_to_np(vis_dict['geometry_detail'][i])))
    imsave(final_out_folder / f"out_im_coarse.png", _fix_image(torch_img_to_np(vis_dict['output_images_coarse'][i])))
    imsave(final_out_folder / f"out_im_detail.png", _fix_image(torch_img_to_np(vis_dict['output_images_detail'][i])))

# 形状コード・表情コードなどをNumpy形式で保存
def save_codes(output_folder, name, vals, i = None):
    if i is None:
        np.save(output_folder / name / f"shape.npy", vals["shapecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"].detach().cpu().numpy())
    else: 
        np.save(output_folder / name / f"shape.npy", vals["shapecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"exp.npy", vals["expcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"tex.npy", vals["texcode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"pose.npy", vals["posecode"][i].detach().cpu().numpy())
        np.save(output_folder / name / f"detail.npy", vals["detailcode"][i].detach().cpu().numpy())

# 単一画像を使用して3D顔復元を行う
def test(deca, img):
    # EMOCAモデルで特徴をエンコード
    vals_encode = deca.encode(img, training=False)
    
    # デバッグ: EMOCAの実際の出力キーを表示
    print(f"EMOCA encode keys: {list(vals_encode.keys())}")
    for k, v in vals_encode.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
    
    # EMOCAの出力をDECA互換のキーにマッピング
    deca_compatible_codes = {}
    
    # EMOCAの実際のキー名に基づいてマッピング
    # 通常のキーマッピング
    key_mapping = {
        'shapecode': 'shape',
        'texcode': 'tex', 
        'expcode': 'exp',
        'posecode': 'pose',
        'lightcode': 'light',
        'detailcode': 'detail',
        'cam': 'cam'
    }
    
    for emoca_key, deca_key in key_mapping.items():
        if emoca_key in vals_encode and vals_encode[emoca_key] is not None:
            deca_compatible_codes[deca_key] = vals_encode[emoca_key]
    
    # 画像を追加
    deca_compatible_codes['images'] = img

    # EMOCAから直接shapeとtexを取得できない場合の対処
    # vals_encodeの中でテンソル型のものを探す
    tensor_keys = {}
    for k, v in vals_encode.items():
        if torch.is_tensor(v):
            tensor_keys[k] = v.shape
    
    print(f"Available tensor keys and shapes: {tensor_keys}")
    
    # shapeとtexが見つからない場合、デフォルト値を生成
    if 'shape' not in deca_compatible_codes:
        # デフォルトのshapeパラメータ（100次元）
        batch_size = img.shape[0]
        deca_compatible_codes['shape'] = torch.zeros(batch_size, 100, device=img.device)
        print(f"Generated default shape: {deca_compatible_codes['shape'].shape}")
    
    if 'tex' not in deca_compatible_codes:
        # デフォルトのtexパラメータ（50次元）
        batch_size = img.shape[0]
        deca_compatible_codes['tex'] = torch.zeros(batch_size, 50, device=img.device)
        print(f"Generated default tex: {deca_compatible_codes['tex'].shape}")

    # identityコードを生成
    if 'shape' in deca_compatible_codes and 'tex' in deca_compatible_codes:
        deca_compatible_codes['identity'] = torch.cat([
            deca_compatible_codes['shape'], 
            deca_compatible_codes['tex']
        ], dim=1)
        print(f"Generated identity code with shape: {deca_compatible_codes['identity'].shape}")
    else:
        print("Warning: Could not generate identity code - missing shape or tex")
        print(f"Available keys: {list(deca_compatible_codes.keys())}")
    
    # 元のvals_encodeに存在した他のキーもコピー
    for k, v in vals_encode.items():
        if k not in deca_compatible_codes and v is not None:
            deca_compatible_codes[k] = v
    
    # デコード処理
    vals, visdict = decode(deca, deca_compatible_codes, training=False)
    
    return deca_compatible_codes, vals, visdict

# 復元された3D顔データの可視化・保存形式への変換
def decode(emoca, values, training=False):
    with torch.no_grad():
        # EMOCAによるデコード、辞書valuesには頂点情報や潜在コードが含まれる
        values = emoca.decode(values, training=training)
        uv_detail_normals = None
        if 'uv_detail_normals' in values.keys():
            uv_detail_normals = values['uv_detail_normals']
        # 可視化用データを生成
        visualizations, grid_image = emoca._visualization_checkpoint(
            values['verts'],        # 顔の3D頂点
            values['trans_verts'],  # カメラ空間に配置された3D頂点
            values['ops'],          # レンダリング情報
            uv_detail_normals,      # UV空間における法線マップ
            values, 
            0,
            "",
            "",
            save=False
        )

    return values, visualizations