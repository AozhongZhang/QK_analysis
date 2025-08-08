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
        H = 8
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
    with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_2k = prompts["2k"]
    inputs = tokenizer(prompt_2k, return_tensors="pt").to(model.device)

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

from scipy.sparse.csgraph import laplacian
import numpy as np

def estimate_n_clusters_by_eigengap(cka_matrix, max_clusters=10):
    L, _ = laplacian(cka_matrix.cpu().numpy(), normed=True, return_diag=True)
    eigvals = np.linalg.eigvalsh(L)
    gaps = np.diff(eigvals[:max_clusters + 1])
    best_k = np.argmax(gaps) + 1
    return best_k

from sklearn.cluster import SpectralClustering

def auto_cluster_heads_from_cka(cka_matrix):
    # Step 1: 估计聚类数量
    n_clusters = estimate_n_clusters_by_eigengap(cka_matrix)
    print("n_clusters: ", n_clusters)
    
    # Step 2: 跑谱聚类
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=0)
    labels = clustering.fit_predict(cka_matrix)
    
    # Step 3: 转成 group_to_rows 格式
    group_to_rows = {}
    for head_id, group_id in enumerate(labels):
        group_to_rows.setdefault(group_id, []).append(head_id)

    return group_to_rows

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np

def auto_cluster_heads(K: torch.Tensor, k_range=(2, 10)):
    """
    自动选择最合适的聚类数量，对每个 head 的表示聚类。
    
    Args:
        K: torch.Tensor of shape [num_heads, seq_len, head_dim]
        k_range: tuple, 聚类数搜索范围
    
    Returns:
        group_to_rows: dict[group_id, list[head_id]]
    """
    num_heads = K.shape[0]
    head_features = K.mean(dim=1).cpu().numpy()  # [num_heads, head_dim]

    best_k = None
    best_score = -1
    best_labels = None

    for k in range(k_range[0], min(k_range[1] + 1, num_heads)):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(head_features)
        score = silhouette_score(head_features, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = kmeans.labels_

    # 输出 group_to_rows
    group_to_rows = {}
    for head_id, group_id in enumerate(best_labels):
        group_to_rows.setdefault(group_id, []).append(head_id)

    return group_to_rows, best_k


def get_group_to_rows_all_layers_by_cka(K_list_all_layers, compute_cka_fn):
    all_group_to_rows = {}
    for layer_idx, K in enumerate(K_list_all_layers):
        cka = compute_cka_fn(K)
        group_to_rows = auto_cluster_heads_from_cka(cka.cpu().numpy())
        all_group_to_rows[layer_idx] = group_to_rows
    return all_group_to_rows


# test
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

K, V = collect_headwise_KV(model, tokenizer, model.device)
print(K[0].shape)

# # result = get_per_layer_group_to_rows(K, compute_similiarity, 0.7)
# result = get_per_layer_group_to_rows_by_cluster_center(K, compute_similiarity)
# # cka_sim = compute_similiarity(K[0])
# # result = group_heads_by_cka_matrix(K[0], cka_sim)
# cka_matrix_K0 = compute_similiarity(K[0])
# print(cka_matrix_K0)
result = auto_cluster_heads(K[0])
# result = auto_cluster_heads_from_cka(cka_matrix_K0)
# result = get_group_to_rows_all_layers_by_cka(K, compute_similiarity)

print(result)