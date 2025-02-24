
<div align="center">
<img src='assets/riflex.png'></img>

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



[Paper](https://arxiv.org/pdf/xxx.xxx.pdf) | [Project Page](https://riflex-video.github.io/) | [Video](https://www.youtube.com/watch?v=taofoXDsKGk) 


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
In the following, we show how to identify the intrinsic frequency:
```python
def identify_k( b: float, d: int, N: int):
    """
    Args:
        b (`float`): The base frequency for RoPE.
        d (`int`): Dimension of the frequency tensor
        N (`int`): the first observed repetition frame in latent space
    Returns:
        k (`int`): the index of intrinsic frequency component
        N_k (`int`): the period of intrinsic frequency component
    """
    # Compute the period of each frequency in RoPE according to Eq.(4)
    periods = []
    for j in range(1, d // 2 + 1):
        theta_j = 1.0 / (b ** (2 * (j - 1) / d))
        N_j = round(2 * torch.pi / theta_j)
        periods.append(N_j)
        print(j, N_j)

    # Identify the intrinsic frequency whose period is closed to N（see Eq.(7)）
    diffs = [abs(N_j - N) for N_j in periods]
    k = diffs.index(min(diffs)) + 1
    N_k = periods[k-1]
    return k, N_k
```
For example, in [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), with `b=256` and `d=16`, the repetition occurs approximately 8s (`N=48` in latent space). In this case, the intrinsic frequency index `k` is 4, and the period `N_k` is 50.

## Inference with Diffusers
#### Installation
```bash
conda create -n riflex python=3.10
pip install -r requirements.txt
pip install -U bitsandbytes
```
#### Single GPU
For training-free 2× temporal extrapolation in HunyuanVideo: 
```bash
python hunyuanvideo.py --k 4 --N_k 50 --num_frames 261 
```
For finetuned 2× temporal extrapolation in HunyuanVideo: 
```bash
python hunyuanvideo.py --k 4 --N_k 66 --num_frames 261 --finetune --model_id "thu-ml/Hunyuan-RIFLEx-diffusers"
```

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