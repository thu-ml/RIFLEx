<div align="center">
<img src='assets/riflex.png'></img>
 <a href='https://arxiv.org/pdf/2502.15894'><img src='https://img.shields.io/badge/arXiv-2502.15894-b31b1b.svg'></a> &nbsp;
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

## Installation
The envrionment is the same with [HunyuanVideo](https://github.com/Tencent/HunyuanVideo).
```bash
# 1. Create conda environment
conda create -n HunyuanVideo python==3.10.9

# 2. Activate the environment
conda activate HunyuanVideo

# 3. Install PyTorch and other dependencies using conda
# For CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.0
```

<details>
<summary> In case of running into `AssertionError: Ulysses Attention and Ring Attention requires xfuser package`, you may update xfusers to 0.4.1 (Click to expand) </summary>
```
pip install xfuser==0.4.1
```
</details>

<details>
<summary>In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions (Click to expand)</summary>

```bash
# Option 1: Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/

# Option 2: Forcing to explictly use the CUDA 11.8 compiled version of Pytorch and all the other packages
pip uninstall -r requirements.txt  # uninstall all packages
pip uninstall -y xfuser
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
pip install xfuser==0.4.0
```
</details>

## Download Models
```shell
cd HunyuanVideo
python -m pip install "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts

# Download our fine-tuned model with RIFLEx
python download.py

# Download text-encoders
cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
python ../hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir llava-llama-3-8b-v1_1-transformers --output_dir text_encoder
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```

## Multi GPU Inference
For training-free 2× temporal extrapolation in HunyuanVideo: 
```bash
torchrun --nproc_per_node=6 sample_video.py \
    --model-base HunyuanVideo/ckpts \
    --dit-weight HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --video-size 544 960 \
    --infer-steps 50 \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 6 \
    --ring-degree 1 \
    --save-path output \
    --prompt "In a forested area, a temporary camp is set up with tents, a dirt ground, and various equipment, including a four-wheeled vehicle and barrels. A man in a white shirt appears distressed, holding his head, while a woman in a brown dress looks on with concern. The presence of military personnel and civilians suggests a situation of conflict or crisis. The mood is tense and somber, with an undercurrent of urgency or the aftermath of a significant event, as evidenced by the body lying on the ground. The camera maintains a steady, medium-long shot, capturing the expressions and movements of the characters, and the realistic, cinematic visual style enhances the gravity of the scene." \
    --k 4 \
    --N_k 50 \
    --video-length 261
```

For finetuned 2× temporal extrapolation in HunyuanVideo: 
```bash
torchrun --nproc_per_node=6 sample_video.py \
    --model-base HunyuanVideo/ckpts \
    --dit-weight HunyuanVideo/ckpts/diffusion_pytorch_model.safetensors \
    --video-size 544 960 \
    --infer-steps 50 \
    --flow-reverse \
    --seed 42 \
    --ulysses-degree 6 \
    --ring-degree 1 \
    --save-path output \
    --prompt "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest." \
    --k 4 \
    --N_k 66 \
    --video-length 261 \
    --finetune
```

The prompts list in the project page are provided in `assets/prompt_free.txt` and `assets/prompt_finetune.txt`.

Note that in the default parallel setting for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), the following number of GPUs are supported:

<details>
<summary>Supported Parallel Configurations (Click to expand)</summary>

| --ulysses-degree x --ring-degree | --nproc_per_node |
|-----------------------------------|------------------|
|  6x1,3x2,2x3,1x6                  | 6                |
|  4x1,2x2,1x4                      | 4                |
|  3x1,1x3                          | 3                |
|  1x2,2x1                          | 2                |

</details>
