from typing import List
import argparse
import torch
from diffusers.utils import export_to_video
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    HunyuanVideoTransformer3DModel,
    HunyuanVideoPipeline
)
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoRotaryPosEmbed
from riflex_utils import get_1d_rotary_pos_embed_riflex

class HunyuanVideoRotaryPosEmbedRifleX(HunyuanVideoRotaryPosEmbed):
    def __init__(self, k: int=4, L_test:int=66, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.L_test = L_test

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(3):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            # === Riflex modification start ===
            # apply Riflex for time dimension
            if i == 0:
                freq = get_1d_rotary_pos_embed_riflex(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True, k=self.k, L_test=self.L_test)
            # === Riflex modification end ===
            else:
                freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)
        return freqs_cos, freqs_sin


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, help='Index of intrinsic frequency', default=4)
    parser.add_argument('--N_k', type=int, help='the period of intrinsic frequency', default=50)
    parser.add_argument('--num_frames', type=int, help='Number of frames for inference', default=261)
    parser.add_argument('--finetune', type=bool, help='whether finetuned version', action='store_true')
    parser.add_argument('--model_id', type=str, help='huggingface path for models', default="hunyuanvideo-community/HunyuanVideo")
    args = parser.parse_args()

    quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
    transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )

    pipe = HunyuanVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        transformer=transformer_8bit,
        torch_dtype=torch.float16,
        device_map="balanced",
    )

    # For training-free, if extrapolated length exceeds the period of intrinsic frequency, modify RoPE
    # For fine-tuning, we finetune the model on RIFLEx so we always modify RoPE
    if args.L_test > args.N_k or args.finetune:
        original_rope = pipe.transformer.rope
        pipe.transformer.rope = HunyuanVideoRotaryPosEmbedRifleX(original_rope.patch_size, original_rope.patch_size_t, original_rope.rope_dim,original_rope.theta, args.k, args.L_test)

    pipe.vae.enable_tiling()

    prompt = "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
    video = pipe(
        prompt=prompt,
        num_frames=args.num_frames,
        num_inference_steps=50,
        height=544,
        width=960,
    ).frames[0]

    export_to_video(video, "demo.mp4", fps=24)


