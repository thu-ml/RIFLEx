import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from hyvideo.modules.attenion import attention
from xfuser.core.long_ctx_attention import xFuserLongContextAttention
from xfuser.core.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    # initialize_runtime_state,
)

def init_dist(backend="nccl"):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(
        f"Initializing distributed environment with rank {rank}, world size {world_size}, local rank {local_rank}"
    )

    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=rank, world_size=world_size)
    # dist.init_process_group(backend=backend)
       # construct a hybrid sequence parallel config (ulysses=2, ring = world_size // 2)

    if world_size > 1:
        ring_degree = world_size // 2
        ulysses_degree = 2
    else:
        ring_degree = 1
        ulysses_degree = 1
    initialize_model_parallel(
        sequence_parallel_degree=world_size,
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
    )

    return rank, world_size

def test_mm_double_stream_block_attention(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    batch_size = 1
    seq_len_img = 118800
    seq_len_txt = 256
    heads_num = 24
    head_dim = 128

    img_q = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    img_k = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    img_v = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
    txt_q = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
    txt_k = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
    txt_v = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)

    with torch.no_grad():
        torch.distributed.broadcast(img_q, src=0)
        torch.distributed.broadcast(img_k, src=0)
        torch.distributed.broadcast(img_v, src=0)
        torch.distributed.broadcast(txt_q, src=0)
        torch.distributed.broadcast(txt_k, src=0)
        torch.distributed.broadcast(txt_v, src=0)
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        

        cu_seqlens_q = torch.tensor([0, 118811, 119056], device='cuda:0', dtype=torch.int32)
        cu_seqlens_kv = torch.tensor([0, 118811, 119056], device='cuda:0', dtype=torch.int32)
        max_seqlen_q = 119056
        max_seqlen_kv = 119056
        mode = "torch" # "torch", "vanilla", "flash"

        original_output = attention(
            q,
            k,
            v,
            mode=mode,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size
        )

        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        hybrid_seq_parallel_output = hybrid_seq_parallel_attn(
            None,
            img_q,
            img_k,
            img_v,
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=txt_q,
            joint_tensor_key=txt_k,
            joint_tensor_value=txt_v,
            joint_strategy="rear",
        )

        b, s, a, d = hybrid_seq_parallel_output.shape
        hybrid_seq_parallel_output = hybrid_seq_parallel_output.reshape(b, s, -1)

        assert original_output.shape == hybrid_seq_parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {hybrid_seq_parallel_output.shape}"

        torch.testing.assert_close(original_output, hybrid_seq_parallel_output, rtol=1e-3, atol=1e-3)
        print("test_mm_double_stream_block_attention Passed")

def test_mm_single_stream_block_attention(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    txt_len = 256
    batch_size = 1
    seq_len_img = 118800
    seq_len_txt = 256
    heads_num = 24
    head_dim = 128

    with torch.no_grad():   
        img_q = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
        img_k = torch.randn(batch_size, seq_len_img, heads_num, head_dim, device=device, dtype=dtype)
        txt_q = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
        txt_k = torch.randn(batch_size, seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len_img + seq_len_txt, heads_num, head_dim, device=device, dtype=dtype)

        torch.distributed.broadcast(img_q, src=0)
        torch.distributed.broadcast(img_k, src=0)
        torch.distributed.broadcast(txt_q, src=0)
        torch.distributed.broadcast(txt_k, src=0)
        torch.distributed.broadcast(v, src=0)

        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)

        cu_seqlens_q = torch.tensor([0, 118811, 119056], device='cuda:0', dtype=torch.int32)
        cu_seqlens_kv = torch.tensor([0, 118811, 119056], device='cuda:0', dtype=torch.int32)
        max_seqlen_q = 119056
        max_seqlen_kv = 119056
        mode = "torch" # "torch", "vanilla", "flash"

        original_output = attention(
            q,
            k,
            v,
            mode=mode,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=batch_size
        )

        hybrid_seq_parallel_attn = xFuserLongContextAttention()
        hybrid_seq_parallel_output = hybrid_seq_parallel_attn(
            None,
            q[:, :-txt_len, :, :],
            k[:, :-txt_len, :, :],
            v[:, :-txt_len, :, :],
            dropout_p=0.0,
            causal=False,
            joint_tensor_query=q[:, -txt_len:, :, :],
            joint_tensor_key=k[:, -txt_len:, :, :],
            joint_tensor_value=v[:, -txt_len:, :, :],
            joint_strategy="rear",
        )
        b, s, a, d = hybrid_seq_parallel_output.shape
        hybrid_seq_parallel_output = hybrid_seq_parallel_output.reshape(b, s, -1)

        assert original_output.shape == hybrid_seq_parallel_output.shape, f"Shape mismatch: {original_output.shape} vs {hybrid_seq_parallel_output.shape}"

        torch.testing.assert_close(original_output, hybrid_seq_parallel_output, rtol=1e-3, atol=1e-3)
        print("test_mm_single_stream_block_attention Passed")

if __name__ == "__main__":
    rank, world_size = init_dist()
    test_mm_double_stream_block_attention(rank, world_size)
    test_mm_single_stream_block_attention(rank, world_size)
