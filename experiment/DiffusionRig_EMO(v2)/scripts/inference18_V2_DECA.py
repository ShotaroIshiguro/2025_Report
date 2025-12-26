import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch as th
from glob import glob

from utils.script_util18_V2 import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from torchvision.utils import save_image
from gdl_apps.EMOCA.utils.load import load_model
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import datasets as deca_dataset

import gdl
from pathlib import Path
from gdl_apps.EMOCA.utils.io import save_obj, save_images, save_codes, test
import pickle

# (LINE Botのコードは省略可能です)

def create_inter_data(dataset, modes, meanshape_path="", target_model=None, source_model=None):
    deca_cfg.model.use_tex = True
    deca_cfg.model.tex_path = "data/FLAME_texture.npz"
    deca_cfg.model.tex_type = "FLAME"
    deca_cfg.rasterizer_type = "pytorch3d"
    deca = DECA(config=deca_cfg)

    EMOCA_path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    EMOCA_model_name = 'EMOCA_v2_lr_mse_20'
    EMOCA_mode = 'detail'
    emoca, conf = load_model(EMOCA_path_to_models, EMOCA_model_name, EMOCA_mode)
    emoca.cuda()
    emoca.eval()

    meanshape = None
    if os.path.exists(meanshape_path):
        print("use meanshape: ", meanshape_path)
        with open(meanshape_path, "rb") as f:
            meanshape = pickle.load(f)
    else:
        print("not use meanshape")

    img2 = dataset[-1]["image"].unsqueeze(0).to("cuda")
    with th.no_grad():
        if target_model == "EMOCA":
            code2, _, _  = test(emoca, img2)
        elif target_model == "DECA":
            code2 = deca.encode(img2)
            # --- 変更箇所：'identity'キーが存在しない場合に作成 ---
            if 'identity' not in code2:
                code2['identity'] = th.cat([code2['shape'], code2['tex']], dim=1)
            # --- 変更ここまで ---
        
    image2 = dataset[-1]["original_image"].unsqueeze(0).to("cuda")

    for i in range(len(dataset) - 1):
        print(str(i+1) + "/" + str(len(dataset)-1) + "枚目生成中・・・")
        img1 = dataset[i]["image"].unsqueeze(0).to("cuda")

        with th.no_grad():
            if source_model == "EMOCA":
                code1, _, _  = test(emoca, img1)
            elif source_model == "DECA":
                code1 = deca.encode(img1)
                # --- 変更箇所：'identity'キーが存在しない場合に作成 ---
                if 'identity' not in code1:
                    code1['identity'] = th.cat([code1['shape'], code1['tex']], dim=1)
                # --- 変更ここまで ---

        ffhq_center = None
        ffhq_center = deca.decode(code1, return_ffhq_center=True)

        tform = dataset[i]["tform"].unsqueeze(0)
        tform = th.inverse(tform).transpose(1, 2).to("cuda")
        original_image = dataset[i]["original_image"].unsqueeze(0).to("cuda")

        code1["tform"] = tform
        if meanshape is not None:
            code1["shape"] = meanshape

        for mode in modes:
            code = {}
            codes_selection = ["shape", "tex", "exp", "pose", "cam", "light", "images", "detail", "tform", "identity"]
            for k in code1:
                if k in codes_selection:
                    code[k] = code1[k].clone()

            if mode == "pose":
                code["pose"][:, :3] = code2["pose"][:, :3]
            elif mode == "light":
                code["light"] = code2["light"]
            elif mode == "exp":
                code["exp"] = code2["exp"]
                code["pose"][:, 3:] = code2["pose"][:, 3:]
            elif mode == "latent":
                pass

            opdict, _ = deca.decode(
                code,
                render_orig=True,
                original_image=original_image,
                tform=code["tform"],
                align_ffhq=True,
                ffhq_center=ffhq_center,
            )

            batch = {}
            batch["image"] = original_image * 2 - 1
            batch["image2"] = image2 * 2 - 1
            batch["rendered"] = opdict["rendered_images"].detach()
            batch["normal"] = opdict["normal_images"].detach()
            batch["albedo"] = opdict["albedo_images"].detach()
            batch["mode"] = mode
            batch["shape"] = code["shape"].detach()
            batch["identity"] = code["identity"].detach()
            yield batch

def main():
    args = create_argparser().parse_args()

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    print(f"Loading model from: {args.model_path}")
    ckpt = th.load(args.model_path)
    model.load_state_dict(ckpt)
    model.to("cuda")
    model.eval()

    imagepath_list = []
    if not os.path.exists(args.source) or not os.path.exists(args.target):
        print("source file or target file doesn't exists.")
        return

    if os.path.isdir(args.source):
        imagepath_list += (
            glob(args.source + "/*.jpg")
            + glob(args.source + "/*.png")
            + glob(args.source + "/*.bmp")
        )
    else:
        imagepath_list += [args.source]
    imagepath_list += [args.target]
    
    dataset = deca_dataset.TestData(imagepath_list, iscrop=True, size=args.image_size)

    modes = args.modes.split(",")
    data = create_inter_data(dataset, modes, args.meanshape, args.target_model, args.source_model)

    if args.sampler == "ddpm":
        sample_fn = diffusion.p_sample_loop
    elif args.sampler == "ddim":
        sample_fn = diffusion.ddim_sample_loop
    elif args.sampler == "dpm_solver":
        from diffusion.samplers.dpm_solver import dpm_solver_sample
        sample_fn = lambda model, shape, noise, clip_denoised, model_kwargs: (
            dpm_solver_sample(
                model,
                diffusion,
                shape,
                noise=noise,
                model_kwargs=model_kwargs,
                steps=args.steps,
                order=2,
                guidance_scale=args.guidance_scale,
                clip_denoised=clip_denoised,
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    noise = th.randn(1, 3, args.image_size, args.image_size).to("cuda")

    vis_dir = args.output_dir
    idx = 0
    for batch in data:
        image = batch["image"]
        image2 = batch["image2"]
        rendered, normal, albedo = batch["rendered"], batch["normal"], batch["albedo"]
        shape = batch["shape"]
        identity = batch["identity"]
        
        physic_cond = th.cat([rendered, normal, albedo], dim=1)

        with th.no_grad():
            if batch["mode"] == "latent":
                detail_cond = model.encode_cond(image2)
            else:
                detail_cond = model.encode_cond(image)
        
        model_kwargs = {
            "physic_cond": physic_cond, 
            "detail_cond": detail_cond,
            "shape": shape,
            "identity": identity,
        }

        sample = sample_fn(
            model,
            (1, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = (sample + 1) / 2.0
        sample = sample.contiguous()

        save_image(
            sample, os.path.join(vis_dir, "{}_".format(idx) + batch["mode"]) + ".png"
        )
        idx += 1


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        use_ddim=True,
        model_path="",
        source="",
        target="",
        output_dir="",
        modes="pose,exp,light",
        meanshape="",
        target_model="",
        source_model="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of sampling steps for dpm‑solver (NFE)."
    )
    parser.add_argument(
        "--sampler", type=str, default="ddim",
        choices=["ddpm", "ddim", "dpm_solver"],
        help="sampling algorithm"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="classifier‑free guidance scale for sampling"
    )

    return parser


if __name__ == "__main__":
    main()
