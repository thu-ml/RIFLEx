
<div align="center">
<img src='assets/riflex.png'></img>


 <a href='https://arxiv.org/abs/xxx.xxx'><img src='https://img.shields.io/badge/arXiv-xxx.xxx-b31b1b.svg'></a> &nbsp;
 <a href='https://riflex-video.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=taofoXDsKGk'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a><br>


<div>
    <a href="https://gracezhao1997.github.io/" target="_blank">Min Zhao</a><sup></sup> | 
    <a href="https://guandehe.github.io/" target="_blank">Guande He</a><sup></sup> | 
    <a href="https://github.com/Chyxx" target="_blank">Yixiao Chen</a><sup></sup> | 
    <a href="https://zhuhz22.github.io/" target="_blank">Hongzhou Zhu</a><sup></sup>|
<a href="https://zhenxuan00.github.io/" target="_blank">Chongxuan Li</a><sup></sup> | 
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml" target="_blank">Jun Zhu</a><sup></sup>
</div>
<div>
    <sup></sup>Tsinghua University
</div>



</div>

---

## RIFLEx Code
RIFLEx only adds a single line of code on the original [1D RoPE](https://github.com/huggingface/diffusers/blob/9c7e205176c30b27c5f44ec7650a8dfcc12dde86/src/diffusers/models/embeddings.py#L1105).
```python
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    '''
        k: the index for the intrinsic frequency in RoPE
        L_test: the number of frames for inference
    '''
    
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)) 

    # === RIFLEx modification start ===
    # Reduce intrinsic frequency to stay within a single period after extrapolation (Eq.(8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply 0.9 to keep extrapolated length below 90% of a period. 
    freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === RIFLEx modification end ===

    freqs = torch.outer(pos, freqs)  
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  
    return freqs_cos, freqs_sin
```
In `riflex_utils.py`, we show how to identify the intrinsic frequency in a RoPE-based pre-trained diffusion transformer.

## Single GPU Inference with Diffusers for Quick Start
### Installation
```bash
conda create -n riflex python=3.10
pip install -r requirements.txt
pip install -U bitsandbytes
```
### Inference
For training-free 2× temporal extrapolation in HunyuanVideo: 
```bash
python hunyuanvideo.py --k 4 --N_k 50 --num_frames 261 --prompt "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
```
For finetuned 2× temporal extrapolation in HunyuanVideo: 
```bash
python hunyuanvideo.py --k 4 --N_k 66 --num_frames 261 --finetune --model_id "thu-ml/Hunyuan-RIFLEx-diffusers" --prompt "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
```
The prompts in the project page are available in `assets/prompt_free.txt` and `assets/prompt_finetune.txt`.

Please note that Diffusers use `DiffusersBitsAndBytesConfig` to save memory, which may impact performance and may not reproduce the demos on the project page.

## Multi GPU Inference (Recommended)
To enhance inference speed and reproduce the demos in our [project page](https://riflex-video.github.io/), please use the multi-gpu inference. Details can be found in the [`multi-gpu` branch](https://github.com/thu-ml/RIFLEx/tree/multi-gpu).


## TODO List
- [x] Release the code and fine-tuned HunyuanVideo for temporal extrapoaltion
- [ ] Support more models (e.g., CogVideoX)
- [ ] Release the code and model for spatial and joint spatial-temporal extrapolation

## References
If you find the code useful, please cite
```
@article{zhao2025RIFLEx,
          title={RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion Transformers}, 
          author={Min Zhao and Guande He and Yixiao Chen and Hongzhou Zhu and Chongxuan Li and Jun Zhu},
          year={2025},
          journal={arXiv:2502.09535},
}
```
