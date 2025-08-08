import os, click
import torch
import torch.nn as nn
from loguru import logger
# from .model import AVAILABLE_MODELS
# from .data_utils import get_calib_data
import math
from tqdm import tqdm
import json


def collect_headwise_KV(model, tokenizer, device):
    
    # calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
    
    # We'll collect outputs for all layers in these lists
    collected_k_outputs = []
    collected_v_outputs = []
    
    def k_proj_hook(module, input, output):
        """
        module: The layer that produced this output (k_proj).
        input:  The input to k_proj.
        output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
        """
        B, S, D = output.shape
        H = 32
        head_dim = D // H
        output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
        # Detach to avoid growing the autograd graph
        collected_k_outputs.append(output.detach().cpu())

    def v_proj_hook(module, input, output):
        """
        module: The layer that produced this output (v_proj).
        input:  The input to v_proj.
        output: The output from v_proj (shape [batch_size, seq_len, hidden_dim]).
        """
        B, S, D = output.shape
        H = 32
        head_dim = D // H
        output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
        # Detach to avoid growing the autograd graph
        collected_v_outputs.append(output.detach().cpu())

    num_layers = len(model.model.layers)
    hooks_k = []
    hooks_v = []
    for layer_idx in range(num_layers):
        # Access the i-th layer
        layer = model.model.layers[layer_idx].self_attn
        # print(f"  - K/V heads: {layer.config.num_key_value_heads}")
        
        
        # Register forward hooks
        hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
        hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
        
        hooks_k.append(hook_k)
        hooks_v.append(hook_v)


    model.eval()
    model.to(device)
    with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama2.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_4k = prompts["4k"]
    inputs = tokenizer(prompt_4k, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    for hook in hooks_k:
        hook.remove()
    for hook in hooks_v:
        hook.remove()
    
    return collected_k_outputs, collected_v_outputs

def linear_cka_centered_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
    """
    A *centered* linear CKA, as in Kornblith et al. (2019), for (L, D) Tensors.
    This subtracts each row's mean from kv1, kv2 before computing the norm-based formula.
    
    Steps:
      1. Row-center each representation (i.e., subtract column means).
      2. Compute Frobenius norms of X^T X, Y^T Y, X^T Y on the centered data.
      3. Return (||X^T Y||_F^2) / (||X^T X||_F * ||Y^T Y||_F).

    Note:
      - 'Row-center' means we subtract the *column* mean for each dimension (the usual approach 
        in CKA references). This ensures the average vector over all tokens is zero.

    Args:
      kv1: shape (L, D)
      kv2: shape (L, D)

    Returns:
      cka_value: a scalar torch.Tensor
    """
    assert kv1.shape[1] == kv2.shape[1], "kv1, kv2 must have same embedding dimension."

    # Move to GPU if desired
    device = kv1.device
    kv1 = kv1.to(device)
    kv2 = kv2.to(device)
    
    # 1. Row-center each representation. 
    #    (Compute column means & subtract => each dimension has mean 0 across L)
    kv1_centered = kv1 - kv1.mean(dim=0, keepdim=True)
    kv2_centered = kv2 - kv2.mean(dim=0, keepdim=True)
    
    # 2. Norm computations
    xtx = (kv1_centered.T @ kv1_centered).norm(p='fro')
    yty = (kv2_centered.T @ kv2_centered).norm(p='fro')
    xty = (kv1_centered.T @ kv2_centered).norm(p='fro')

    # Handle degenerate case
    if xtx == 0 or yty == 0:
        return torch.tensor(0.0, device=device, dtype=kv1.dtype)

    # 3. Linear CKA formula
    cka_value = (xty ** 2) / (xtx * yty)

    return cka_value

def compute_similiarity(matrix):
    
    num_heads, seq_len, head_dim = matrix.shape
    cka_matrix = torch.zeros(num_heads, num_heads)
    cka_matrix = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        for j in range(num_heads):
            vi = matrix[i]
            vj = matrix[j]
            assert vi.shape == vj.shape
            vi = vi.cuda().float()
            vj = vj.cuda().float()
    
            cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
            
    return cka_matrix


import torch
import numpy as np
from typing import Dict, List, Tuple

def greedy_group_by_cka(
    cka_matrix: torch.Tensor,
    group_size: int = 4,
    seed: int = 42
) -> Tuple[Dict[int, List[int]], List[int]]:
    """
    使用贪心算法基于 CKA 相似度将 head 重新排序并划分 group。
    """
    rng = np.random.default_rng(seed)
    cka = cka_matrix.cpu().numpy()
    num_heads = cka.shape[0]

    remaining = set(range(num_heads))
    group_to_rows: Dict[int, List[int]] = {}
    # order: List[int] = []

    gid = 0
    while remaining:
        # 1. 选一个随机种子 head
        seed_head = rng.choice(list(remaining))
        remaining.remove(seed_head)

        # 2. 找出与 seed 最相似的 (group_size - 1) 个 head
        sim = cka[seed_head].copy()
        sim[list(set(range(num_heads)) - remaining - {seed_head})] = -np.inf
        top_indices = np.argsort(sim)[- (group_size - 1):][::-1]
        top_heads = []

        for idx in top_indices:
            if idx in remaining:
                top_heads.append(idx)
                remaining.remove(idx)
            if len(top_heads) == group_size - 1:
                break

        # 3. 如果还不够，继续从剩下里随机补
        while len(top_heads) < group_size - 1 and remaining:
            extra = rng.choice(list(remaining))
            top_heads.append(extra)
            remaining.remove(extra)

        # 4. 组成组并记录
        group = [seed_head] + top_heads
        group_to_rows[gid] = group
        # order.extend(group)
        gid += 1

    return group_to_rows


def get_group_to_rows_all_layers_by_cka(K_list_all_layers, compute_cka_fn):
    all_group_to_rows = {}
    for layer_idx, K in enumerate(K_list_all_layers):
        cka = compute_cka_fn(K)
        group_to_rows = greedy_group_by_cka(cka, group_size=8)
        all_group_to_rows[layer_idx] = group_to_rows
    return all_group_to_rows


def build_ordered_group_to_rows(K_list_all_layers, group_size=4):
    """
    Args:
        K_list_all_layers: List of tensors, each with shape [num_head, seq_len, head_dim]
        group_size: fixed size for each group

    Returns:
        group_to_rows_all_layers: Dict[int, Dict[int, List[int]]], layer_idx -> group_id -> list of head idx
    """
    group_to_rows_all_layers = {}

    for layer_idx, K in enumerate(K_list_all_layers):
        num_heads = K.shape[0]
        group_to_rows = {}
        gid = 0
        for i in range(0, num_heads, group_size):
            group_to_rows[gid] = list(range(i, min(i + group_size, num_heads)))
            gid += 1
        group_to_rows_all_layers[layer_idx] = group_to_rows

    return group_to_rows_all_layers

# # test
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# K, V = collect_headwise_KV(model, tokenizer, model.device)
# result = build_ordered_group_to_rows(K)
# print(result)
# print(K[0].shape)

# # result = get_per_layer_group_to_rows(K, compute_similiarity, 0.7)
# result = get_per_layer_group_to_rows_by_cluster_center(K, compute_similiarity)
# # cka_sim = compute_similiarity(K[0])
# # result = group_heads_by_cka_matrix(K[0], cka_sim)
# cka_matrix_K0 = compute_similiarity(K[0])
# print(cka_matrix_K0)
# result, order = greedy_group_by_cka(cka_matrix_K0)
# result = get_group_to_rows_all_layers_by_cka(K, compute_similiarity)
# result, order = reorder_heads_by_cka(cka_matrix_K0, 4)
# print(cka_matrix_K0)
# result = auto_cluster_heads(K[0])
# result = auto_cluster_heads_from_cka(cka_matrix_K0)
# result = get_group_to_rows_all_layers_by_cka(K, compute_similiarity)

# print(result)
# print(order)

# import matplotlib.pyplot as plt
# import seaborn as sns

# reordered = cka_matrix_K0[order][:, order]
# plt.figure(figsize=(6, 5))
# sns.heatmap(reordered.cpu().numpy(), cmap="coolwarm", vmin=0, vmax=1)
# plt.title("CKA Similarity after Reordering")
# plt.xlabel("Head Index")
# plt.ylabel("Head Index")
# plt.grid(True)
# plt.tight_layout()

    
# plt.savefig(f"/home/azzhang/Palu/palu/raw_rope.png")
# plt.show()

# def plot_reordered_cka(cka_matrix, order):
#     reordered = cka_matrix[order][:, order]
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(reordered.cpu().numpy(), cmap="coolwarm", vmin=0, vmax=1)
#     plt.title("CKA Similarity after Reordering")
#     plt.xlabel("Head Index")
#     plt.ylabel("Head Index")
#     plt.show()