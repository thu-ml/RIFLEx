
<div align="center">

<h1>RIFLEx: A Free Lunch for Length Extrapolation in
Video Diffusion Transformers</h1>

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

```python
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)) 

    # === Riflex modification start ===
    # Reduce intrinsic frequency to stay within a single period after extrapolation (Eq.(8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply 0.9 to keep extrapolated length below 90% of a period.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  
    return freqs_cos, freqs_sin
```