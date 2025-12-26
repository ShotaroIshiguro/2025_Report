import copy
import functools
import os
import time
import datetime

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from . import dist_util, logger
# --- 変更箇所：インポートパスを修正 ---
from diffusion.resample import LossAwareSampler, UniformSampler
# --- 変更ここまで ---

# (LINE Botのコードは省略可能です)

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        stage=1,
        max_steps=0,
        auto_scale_grad_clip=1.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.stage = stage
        self.max_steps = max_steps
        self.auto_scale_grad_clip = auto_scale_grad_clip

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()

        # MixedPrecisionTrainer or GradScaler setup
        # Note: The original file seems to have a custom MixedPrecisionTrainer, 
        # but the error log implies torch.cuda.amp.GradScaler is used.
        # We will proceed with GradScaler as it's more standard.
        self.scaler = GradScaler()

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()

        self.ddp_model = DDP(
            self.model,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False,
        )

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.max_steps or self.step < self.max_steps
        ):
            batch = next(self.data)
            self.run_step(batch)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", ""):
                    return
            self.step += 1
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        self.scaler.unscale_(self.opt)
        th.nn.utils.clip_grad_norm_(self.model.parameters(), self.auto_scale_grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.log_step()

    def forward_backward(self, batch):
        self.opt.zero_grad()
        
        micro_image = batch["image"].to(dist_util.dev())
        micro_rendered = batch["rendered"].to(dist_util.dev())
        micro_normal = batch["normal"].to(dist_util.dev())
        micro_albedo = batch["albedo"].to(dist_util.dev())
        micro_shape = batch["shape"].to(dist_util.dev())
        micro_identity = batch["identity"].to(dist_util.dev())
        
        micro_physic_cond = th.cat([micro_rendered, micro_normal, micro_albedo], dim=1)
        
        t, weights = self.schedule_sampler.sample(micro_image.shape[0], dist_util.dev())

        model_kwargs = dict(
            physic_cond=micro_physic_cond,
            shape=micro_shape,
            identity=micro_identity,
            x_start=micro_image, # x_start is the original image
        )
        
        with autocast():
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro_image, # x_start
                t,
                model_kwargs=model_kwargs,
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        self.scaler.scale(loss).backward()

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(params):
            state_dict = self.model.state_dict()
            if dist.get_rank() == 0:
                logger.log(f"saving model...")
                filename = f"model{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(self.model.parameters())

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
