import argparse
from typing import *
from types import MethodType
from functools import partial
import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel
)

from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.utils import export_to_video
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from riflex_utils import get_1d_rotary_pos_embed_riflex


def get_3d_rotary_pos_embed_riflex(
        embed_dim=None,
        crops_coords=None,
        grid_size=None,
        temporal_size=None,
        theta: int = 10000,
        use_real: bool = True,
        grid_type: str = "linspace",
        max_size: Optional[Tuple[int, int]] = None,
        device: Optional[torch.device] = None,
        k: int = None,
        L_test: int = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    This function is copied from https://github.com/huggingface/diffusers/blob/7007febae5cff000d4df9059d9cf35133e8b2ca9/src/diffusers/models/embeddings.py#L816
    and we only modify line 91 to apply RIFLEx.

    RIFLEx for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.
    k (`int`, *optional*, defaults to None): the index for the intrinsic frequency in RoPE
    L_test (`int`, *optional*, defaults to None): the number of frames for inference

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """

    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.linspace(
            start[0], stop[0] * (grid_size_h - 1) / grid_size_h, grid_size_h, device=device, dtype=torch.float32
        )
        grid_w = torch.linspace(
            start[1], stop[1] * (grid_size_w - 1) / grid_size_w, grid_size_w, device=device, dtype=torch.float32
        )
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        grid_t = torch.linspace(
            0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, device=device, dtype=torch.float32
        )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # === RIFLEx modification start ===
    # Apply RIFLEx for time dimension
    freqs_t = get_1d_rotary_pos_embed_riflex(dim_t, grid_t, theta=theta, use_real=True, k=k, L_test=L_test)
    # === RIFLEx modification end ===

    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


def _prepare_rotary_positional_embeddings_riflex(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
        k: int = None,
        L_test: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    This function is copied from https://github.com/huggingface/diffusers/blob/7007febae5cff000d4df9059d9cf35133e8b2ca9/src/diffusers/pipelines/cogvideo/pipeline_cogvideox.py#L441
    and we only modify line 160 and line 175 to apply RIFLEx.
    '''
    grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
    grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

    p = self.transformer.config.patch_size
    p_t = self.transformer.config.patch_size_t

    base_size_width = self.transformer.config.sample_width // p
    base_size_height = self.transformer.config.sample_height // p

    if p_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        # === RIFLEx modification start ===
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed_riflex(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
            k=k,
            L_test=L_test,
        )
        # === RIFLEx modification end ===
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + p_t - 1) // p_t

        # === RIFLEx modification start ===
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed_riflex(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
            device=device,
        )
        # === RIFLEx modification end ===

    return freqs_cos, freqs_sin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed', default=1234)
    parser.add_argument('--k', type=int, help='Index of intrinsic frequency', default=2)
    parser.add_argument('--N_k', type=int, help='The period of intrinsic frequency in latent space', default=20)
    parser.add_argument('--num_frames', type=int, help='Number of frames for inference', default=97)
    parser.add_argument('--finetune', help='Whether finetuned version', action='store_true')
    parser.add_argument('--model_id', type=str, help='huggingface path for models', default="THUDM/CogVideoX-5b")
    parser.add_argument('--prompt', type=str, help='Prompts for generation',
                        default="3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest.")
    args = parser.parse_args()

    assert (args.num_frames - 1) % 4 == 0, "num_frames should be 4 * k + 1"
    L_test = (args.num_frames - 1) // 4 + 1  # latent frames
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator("cuda").manual_seed(args.seed)

    # For training-free, if extrapolate length exceeds the period of intrinsic frequency, modify RoPE
    if L_test > args.N_k and not args.finetune:
        pipe._prepare_rotary_positional_embeddings = MethodType(
            partial(_prepare_rotary_positional_embeddings_riflex, k=args.k, L_test=L_test), pipe)

    # We fine-tune the model on new theta_k and N_k, and thus modify RoPE to match the fine-tuning setting.
    if args.finetune:
        L_test = args.N_k  # the fine-tuning frequency setting
        pipe._prepare_rotary_positional_embeddings = MethodType(
            partial(_prepare_rotary_positional_embeddings_riflex, k=args.k, L_test=L_test), pipe)

    video = pipe(prompt=args.prompt, num_frames=args.num_frames, height=480, width=720, guidance_scale=6,
                 num_inference_steps=50, generator=generator).frames[0]
    export_to_video(video, f"seed_{args.seed}_{args.prompt[:20]}.mp4", fps=8)