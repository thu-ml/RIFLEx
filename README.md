
<div align="center">

<h1>FreeU: Free Lunch in Diffusion U-Net</h1>

<div>
    <a href="https://chenyangsi.github.io/" target="_blank">Chenyang Si</a><sup></sup> | 
    <a href="https://ziqihuangg.github.io/" target="_blank">Ziqi Huang</a><sup></sup> | 
    <a href="https://yumingj.github.io/" target="_blank">Yuming Jiang</a><sup></sup> | 
    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup></sup>
</div>
<div>
    <sup></sup>S-Lab, Nanyang Technological University
</div>



[Paper](https://arxiv.org/pdf/2309.11497.pdf) | [Project Page](https://chenyangsi.top/FreeU/) | [Video](https://www.youtube.com/watch?v=-CZ5uWxvX30&t=2s) | [Demo](https://huggingface.co/spaces/ChenyangSi/FreeU)


<div>
    <sup></sup>CVPR2024 Oral
</div>




    
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Us)](https://twitter.com/scy994)
![](https://img.shields.io/github/stars/ChenyangSi/FreeU?style=social)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FChenyangSi%2FFreeU&count_bg=%23E5970E&title_bg=%23847878&icon=&icon_color=%23E7E7E7&title=Github+visitors&edge_flat=false)](https://hits.seeyoufarm.com)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fhuggingface.co%2Fspaces%2FChenyangSi%2FFreeU&count_bg=%23E5D10E&title_bg=%23847878&icon=&icon_color=%23E7E7E7&title=HuggingFace+visitors&edge_flat=false)](https://hits.seeyoufarm.com)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fchenyangsi.top%2FFreeU%2F&count_bg=%239016D2&title_bg=%23847878&icon=&icon_color=%23E7E7E7&title=Page+visitors&edge_flat=false)](https://hits.seeyoufarm.com)
[![Hugging Face](https://img.shields.io/badge/Demo-%F0%9F%A4%97%20Hugging%20Face-66cdaa)](https://huggingface.co/spaces/ChenyangSi/FreeU)

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
    """
    RIFLEx: Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)) 

     
    # === Riflex modification start ===
    # Reduce intrinsic frequency to stay within a single cycle after extrapolation (see Eq.(8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply 0.9 to keep extrapolated length below 90% of a single cycle.
    if k is not None:
        freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === Riflex modification end ===

    freqs = torch.outer(pos, freqs)  
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  
    return freqs_cos, freqs_sin
    
```

identifying_k.py :

HunyuanVideo: `python identifying_k.py --b 256 --d 16 --len 66`

CogVideo: ` python identifying_k.py --b 10000 --d 16 --len 26`

参数含义见 args 的 help