import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

def merge_safetensors(part1_path, part2_path, merged_path):
    part1 = load_file(part1_path)
    part2 = load_file(part2_path)
    
    merged = {}
    for k in list(part1.keys()) + list(part2.keys()):
        merged[k] = part1[k] if k in part1 else part2[k]
    
    save_file(merged, merged_path)

if __name__ == "__main__":

    os.makedirs("ckpts", exist_ok=True)
    os.chdir("ckpts") 


    repo_id = "thu-ml/Hunyuan-RIFLEx"
    
    slice1 = "slice1.safetensors"
    hf_hub_download(
        repo_id=repo_id,
        filename=slice1,
        local_dir=".",
        repo_type="model"
    )

    slice2 = "slice2.safetensors"
    hf_hub_download(
        repo_id=repo_id,
        filename=slice2,
        local_dir=".",
        repo_type="model"
    )

    merged_file = "diffusion_pytorch_model.safetensors"
    merge_safetensors(slice1, slice2, merged_file)
    os.remove(slice1)
    os.remove(slice2)
    print("Model has been downloaded to `ckpts/diffusion_pytorch_model.safetensors`.")