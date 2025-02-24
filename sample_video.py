import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
from types import MethodType
from functools import partial

import torch
from HunyuanVideo.hyvideo.utils.file_utils import save_videos_grid
from HunyuanVideo.hyvideo.config import parse_args
from HunyuanVideo.hyvideo.inference import HunyuanVideoSampler
from HunyuanVideo.hyvideo.modules.posemb_layers import get_meshgrid_nd
from riflex_utils import get_1d_rotary_pos_embed_riflex
from diffusers.models.embeddings import get_1d_rotary_pos_embed


def get_nd_rotary_pos_embed_riflex(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    k: int = None,
    L_test: int = None
):
    """
    This function is copied from https://github.com/Tencent/HunyuanVideo/blob/ed32900b2ece5f7f42a0a718384f3a54fade33e3/hyvideo/modules/posemb_layers.py#L191
    and we only modify line 59 to apply RIFLEx.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.
        k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
        L_test (`int`, *optional*, defaults to None): the number of frames for inference

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        if i == 0:
            # === RIFLEx modification start ===
            # Apply Riflex for the temporal dimension
            emb = get_1d_rotary_pos_embed_riflex(
                rope_dim_list[i],
                grid[i].reshape(-1),
                theta,
                use_real=use_real,
                k=k,
                L_test=L_test
            )
            embs.append(emb)
            # === Riflex modification end ===
        else:
            emb = get_1d_rotary_pos_embed(
                rope_dim_list[i],
                grid[i].reshape(-1),
                theta,
                use_real=use_real,
            ) 
            embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb

    
def get_rotary_pos_embed(self, video_length, height, width, k, L_test):
    """
    This function is copied from https://github.com/Tencent/HunyuanVideo/blob/ed32900b2ece5f7f42a0a718384f3a54fade33e3/hyvideo/inference.py#L450
    and we only modify line 131 to apply RIFLEx.
    """
    target_ndim = 3
    ndim = 5 - 2
    # 884
    if "884" in self.args.vae:
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
    elif "888" in self.args.vae:
        latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
    else:
        latents_size = [video_length, height // 8, width // 8]

    if isinstance(self.model.patch_size, int):
        assert all(s % self.model.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // self.model.patch_size for s in latents_size]
    elif isinstance(self.model.patch_size, list):
        assert all(
            s % self.model.patch_size[idx] == 0
            for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({self.model.patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [
            s // self.model.patch_size[idx] for idx, s in enumerate(latents_size)
        ]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = self.model.hidden_size // self.model.heads_num
    rope_dim_list = self.model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    
    # === RIFLEx modification start ===
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed_riflex(
        rope_dim_list=rope_dim_list,
        start=rope_sizes,
        theta=self.args.rope_theta,
        use_real=True,
        k=k,
        L_test=L_test
    )
    # === RIFLEx modification end ===
    
    return freqs_cos, freqs_sin
    
    
    
def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # === RIFLEx modification start ===
    assert (args.video_length-1) % 4 == 0, "num_frames should be 4 * k + 1"
    L_test = (args.video_length-1) // 4 + 1
    # For training-free, if extrapolate length exceeds the period of intrinsic frequency, modify RoPE
    if L_test > args.N_k and not args.finetune:
        hunyuan_video_sampler.get_rotary_pos_embed = MethodType(partial(get_rotary_pos_embed, k=args.k, L_test=L_test), hunyuan_video_sampler)
    
    # We fine-tune the model on new theta_k and N_k, and thus modify RoPE to match the fine-tuning setting.
    if args.finetune:
        L_test = args.N_k
        hunyuan_video_sampler.get_rotary_pos_embed = MethodType(partial(get_rotary_pos_embed, k=args.k, L_test=L_test), hunyuan_video_sampler)
    # === RIFLEx modification end ===
    

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
