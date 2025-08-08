import torch
import torch.nn as nn
from .quant import Quantizer
from .hadamard_utils import apply_hadamard
from .cka_utils import compute_similiarity, greedy_max_avg_cka_groups, greedy_group_by_cka, compute_cosine_similarity

def _per_head_whiten_decomposition_from_weight(weight, scaling_diag_matrix, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def _per_head_whiten_decomposition_from_weight_fisher_aware(weight, scaling_diag_matrix, fisher_info, rank):
    original_dtype = weight.dtype
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(weight.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")
    
    fisher_info = fisher_info.to(torch.float32).to(weight.device)
    
    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(weight.to(torch.float32), scaling_diag_matrix.to(torch.float32))
    
    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)
    
    V = torch.matmul(Vt, scaling_matrix_inv)

    # Compute Fisher-aware scores per direction
    scores = []
    for i in range(U.shape[1]):
        score = (U[:, i].T @ fisher_info @ V[i, :].T).item() ** 2
        scores.append((score, i))
        # rank-1 outer product: [D_out, D_in]
        # outer = torch.ger(U[:, i], V[i, :])  # u_i ⊗ v_i^T
        # score = (fisher_info * outer).pow(2).sum().item()
        # scores.append((score, i))

    # Select top-k Fisher-aware directions
    scores.sort(reverse=True)
    top_idx = [i for _, i in scores[:rank]]

    # Construct low-rank approximation
    U = U[:, top_idx]  # [D_out, r]
    S = S[top_idx]     # [r]
    V = V[top_idx, :]  # [r, D_in]
    
    # # Low rank approximation to the target rank
    # U = U[:, :rank]
    # S = S[:rank]
    # V = V[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)
    
    return L, R

def fisher_aligned_pca_rank_selection(W: torch.Tensor, fisher: torch.Tensor, rank: int, score_mode: str = "wf"):
    """
    在 W 和 Fisher 的联合主轴空间中，选择最重要的方向来构建 W ≈ L @ R。

    Args:
        W: 权重矩阵 [D_out, D_in]
        fisher: Fisher 信息矩阵 [D_out, D_in]
        rank: 目标保留秩
        score_mode: 排序策略
            - "w": 只看 W 的能量 |⟨W, v_i⟩|^2
            - "wf": Fisher 加权方向 |⟨W, v_i⟩ * ⟨F, v_i⟩|
            - "w+f": 加权能量和

    Returns:
        L: [D_out, rank]
        R: [rank, D_in]
        top_scores: [rank] 方向打分（可用于调试）
    """
    original_dtype = W.dtype
    D_out, D_in = W.shape
    D = D_out * D_in

    W_flat = W.to(torch.float32).view(1, -1)
    F_flat = fisher.to(torch.float32).view(1, -1)

    # Step 1: 构造共享方向空间 basis
    Z = torch.cat([W_flat, F_flat], dim=0)  # shape: [2, D]
    Z_centered = Z - Z.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Z_centered, full_matrices=False)  # Vh: [2, D]
    basis = Vh  # 每行一个方向

    # num_dir = min(rank * 4, basis.shape[0])  # 多保留点候选方向
    # basis = basis[:num_dir, :]  # shape: [num_dir, D]

    # Step 2: 将 W 和 F 同时投影
    W_proj = (W_flat @ basis.T).squeeze(0)  # [num_dir]
    F_proj = (F_flat @ basis.T).squeeze(0)  # [num_dir]
    print(W_proj.shape)
    # Step 3: 排序得分
    if score_mode == "w":
        scores = W_proj.pow(2)
    elif score_mode == "wf":
        scores = (W_proj * F_proj).abs()
    elif score_mode == "w+f":
        scores = W_proj.pow(2) + F_proj.pow(2)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")
    print(len(scores))
    top_idx = torch.topk(scores, k=rank).indices  # [rank]
    top_scores = scores[top_idx]
    V_select = basis[top_idx, :]  # [rank, D]

    # Step 4: 重建低秩矩阵并 SVD 拆分为 L @ R
    W_proj = (W_flat @ V_select.T)           # [1, rank]
    W_recon_flat = W_proj @ V_select         # [1, D]
    W_recon = W_recon_flat.view(D_out, D_in)

    U, S, Vh = torch.linalg.svd(W_recon, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    V = Vh[:rank, :]
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, V).to(original_dtype)

    return L, R



# import torch

# def fisher_aware_svd_scoring(W: torch.Tensor, F: torch.Tensor, rank: int):
#     """
#     对 Linear 层的权重矩阵 W 做 SVD，并使用 Fisher 信息 F 给出每个奇异方向的重要性打分。
    
#     Args:
#         W: 权重矩阵 [out, in]
#         F: Fisher 信息矩阵 [out, in]（每个权重的重要性）
#         rank: 希望保留的方向数

#     Returns:
#         top_u, top_s, top_v: Fisher-aware top-r 重建用的 U, S, V
#     """
#     assert W.shape == F.shape, "Fisher must match weight shape"
    
#     U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh: [rank, in]
#     V = Vh  # already transposed in new PyTorch

#     scores = []
#     for i in range(U.shape[1]):
#         outer = torch.ger(U[:, i], V[i, :])  # outer product: [out, in]
#         score = (F * outer).pow(2).sum().item()
#         scores.append((score, i))

#     # 排序，保留最重要的 top-r
#     scores.sort(reverse=True)
#     top_idx = [i for _, i in scores[:rank]]

#     U_r = U[:, top_idx]  # [out, r]
#     S_r = S[top_idx]     # [r]
#     V_r = V[top_idx, :]  # [r, in]

#     return U_r, S_r, V_r

import torch

def fisher_weighted_svd(W: torch.Tensor, F: torch.Tensor, rank: int):
    """
    用 Fisher 权重打分的 SVD 压缩方法。
    
    Args:
        W: 权重矩阵 [out, in]
        F: Fisher 信息矩阵 [out, in]
        rank: 需要保留的 rank
    
    Returns:
        U_r, S_r, V_r: Fisher-aware top-r 分量
    """
    assert W.shape == F.shape

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    V = Vh  # shape: [r, in]

    scores = []
    for i in range(U.shape[1]):
        score = (U[:, i].T @ F @ V[i, :].T).item() ** 2
        scores.append((score, i))

    # 根据打分排序
    scores.sort(reverse=True)
    top_idx = [i for _, i in scores[:rank]]

    U_r = U[:, top_idx]
    S_r = S[top_idx]
    V_r = V[top_idx, :]

    return U_r, S_r, V_r

def _per_head_decomposition_from_weight_fisher_aware(weight, F, rank):
    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    scores = []
    for i in range(U.shape[1]):
        score = (U[:, i].T @ F @ Vt[i, :].T).item() ** 2
        scores.append((score, i))

    # 根据打分排序
    scores.sort(reverse=True)
    top_idx = [i for _, i in scores[:rank]]

    U = U[:, top_idx]
    S = S[top_idx]
    Vt = Vt[top_idx, :]

    # # Low rank approximation to the target rank
    # U = U[:, :rank]
    # S = S[:rank]
    # Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(original_dtype)
    # assert torch.allclose(torch.matmul(L, R), weight, atol=1e-3), "SVD decomposition failed"
    return L, R

def _per_head_decomposition_from_weight(weight, rank):
    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    sqrtSigma = torch.sqrt(torch.diag(S))
    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma).to(original_dtype)
    R = torch.matmul(sqrtSigma, Vt).to(original_dtype)
    assert torch.allclose(torch.matmul(L, R), weight, atol=1e-3), "SVD decomposition failed"
    return L, R


def _per_head_decomposition_from_weight_calibration(H, M, rank):
    
    r = rank
    
    # U_H, Sigma_H, V_H = torch.svd(H)
    U_H, Sigma_H, Vt_H = torch.linalg.svd(H, full_matrices=False)
    Sigma_H_sqrt = torch.sqrt(Sigma_H) 
    X_0 = Sigma_H_sqrt.unsqueeze(1) * Vt_H
    # X_0 = Vt_H
    # X_0 = Sigma_H_sqrt.unsqueeze(1) * V_H.T
    
    X_0M = X_0 @ M
    # U, Sigma, V = torch.svd(X_0M)
    U, Sigma, Vt = torch.linalg.svd(X_0M, full_matrices=False)
    
    U_r = U[:, :r]
    Sigma_r = torch.diag(Sigma[:r])
    # V_r = V[:, :r]
    Vt_r = Vt[:r, :]
    
    # A = V_H @ torch.diag(1.0 / Sigma_H_sqrt) @ (U_r @ Sigma_r)
    # B = V_r.T
    # A = Vt_H.T @ torch.diag(1.0 / Sigma_H_sqrt) @ (U_r @ Sigma_r)
    # B = Vt_r

    A = Vt_H.T @ torch.diag(1.0 / Sigma_H_sqrt) @ (U_r @ torch.sqrt(Sigma_r))
    B = torch.sqrt(Sigma_r) @ Vt_r
    
    return B.T, A.T
    # return A, B


class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(self, ranks, inverse_indices, in_features, out_features, bias):
        super().__init__()

        self.ranks = ranks
        self.inverse_indices = inverse_indices
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        self.group_dim = out_features // self.num_groups
        

        if (self.group_dim * self.num_groups) != self.out_features:
            raise ValueError(
                f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
                f" and `num_groups`: {self.num_groups})."
            )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)

        Us = []
        for r in ranks:
            Us.append(nn.Linear(r, self.group_dim, bias=bias))

        self.U = nn.ModuleList(Us) 

        self.quantized_latents = False
        self.latent_quantizer = None

    def forward(self, hidden_states: torch.Tensor):
        latents = self.project_to_latent(hidden_states)
        if self.quantized_latents:
            latents = self.quantize_latent(latents)
        
        # return self.reconstruct(latents)
        return self.reconstruct_inverse(latents)

    def project_to_latent(self, hidden_states: torch.Tensor):
        return self.VT(hidden_states)

    def reconstruct(self, latents: torch.Tensor):
        outputs = []
        offset = 0
        for i, r in enumerate(self.ranks):
            out = self.U[i](latents[:, :, offset:offset + r])
            outputs.append(out)
            offset += r
        return torch.cat(outputs, dim=-1)
    
    def reconstruct_inverse(self, latents: torch.Tensor):
        outputs = []
        offset = 0
        for i, r in enumerate(self.ranks):
            out = self.U[i](latents[:, :, offset:offset + r])
            outputs.append(out)
            offset += r
        output = torch.cat(outputs, dim=-1) # [bsz, seq_len, out_feature]

        if self.inverse_indices is not None:
            B, L, D = output.shape
            num_heads = len(self.inverse_indices)
            head_dim = D // num_heads
            output = output.view(B, L, num_heads, head_dim)
            output = output[:, :, self.inverse_indices, :]
            output = output.reshape(B, L, D)

        return output
    
    def quantize_latent(self, latents: torch.Tensor):
        assert self.latent_quantizer is not None
        outputs = []
        offset = 0
        for i, r in enumerate(self.ranks):
            q = self.latent_quantizer(latents[:, :, offset:offset + r])
            outputs.append(q)
            offset += r
        return torch.cat(outputs, dim=-1)

    @staticmethod
    def from_linear_whiten(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, None, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group_head, head_dim, in_feature]
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module
    
    @staticmethod
    def from_linear_whiten_fisher_aware(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, None, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group_head, head_dim, in_feature]
        print("old_module.fisher_info.shape: ", old_module.fisher_info.shape)
        fisher_info = old_module.fisher_info.reshape(len(ranks), -1, old_module.in_features)
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            # l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            l, r = fisher_aligned_pca_rank_selection(w[i], fisher_info[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear_whiten_simreorder(
        old_module: nn.Linear,
        ranks: list,
    ):   
        # new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        # w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group_head, head_dim, in_feature]
        
        w_ = old_module.weight.data.reshape(32, -1, old_module.in_features) # [num_head, head_dim, in_feature]
        num_head, head_dim, in_features = w_.shape
        
        # compute CKA
        # cka_w = compute_similiarity(w_)
        cos_w = compute_cosine_similarity(w_)
        # print(cka_w.shape)
        # grouping by cka
        groups = greedy_group_by_cka(cos_w)
        print(groups)
        # groups = greedy_max_avg_cka_groups(cka_w)
        reorder_indices = [idx for group in groups for idx in group]
        
        # inverse: original head → reordered index
        inverse_indices = [0] * len(reorder_indices)
        for i, j in enumerate(reorder_indices):
            inverse_indices[j] = i

        new_module = HeadwiseLowRankModule(ranks, inverse_indices, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        
        w_reordered = w_[reorder_indices].reshape(len(ranks), -1, old_module.in_features)

        if len(groups) != len(ranks):
            raise ValueError(f"{len(groups)} != {len(ranks)}")
        
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(num_head, head_dim)
            b = b[reorder_indices]
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_whiten_decomposition_from_weight(w_reordered[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (group_dim, rank), r: (rank, in_feature)
            wl.append(l)
            wr.append(r)

        # concat wl together: [num_head, head_dim, ranks]
        # wl_concat = torch.cat(wl, dim=0)
        
        # # out_feature, _ = wl_concat.shape
        # wl_concat = wl_concat.view(len(reorder_indices), head_dim, -1) # [num_head, head_dim, ranks]
        
        # wl_reorder_back = wl_concat[inverse_indices]
        # wl_reorder_back = wl_reorder_back.reshape(len(ranks), (num_head*head_dim)//len(ranks), -1)  # [num_head, head_dim, rank]
        # print("wl_reorder_back_i: ", wl_reorder_back[0].shape)
        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()

        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module

    @staticmethod
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):
        new_module = HeadwiseLowRankModule(ranks, None, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features)
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        wl = []
        wr = []
        for i in range(len(ranks)):
            l, r = _per_head_decomposition_from_weight(w[i], ranks[i])
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]
        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module
    
    @staticmethod
    def from_linear_cloq(
        old_module: nn.Linear,
        ranks: list,
    ):   
        new_module = HeadwiseLowRankModule(ranks, None, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group_head, head_dim, in_feature]
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):
            # l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            l, r = _per_head_decomposition_from_weight_calibration(old_module.scaling_diag_matrix.to(w.device), w[i].T.to(torch.float32), ranks[i])
            
            # l: (head_dim, rank), r: (rank, hidden_size)
            wl.append(l)
            wr.append(r)

        # load to U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].contiguous()
            # Handle the cases where the bias is not None
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # load to VT
        # shape (sum(ranks), hidden_size)
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight
        
        return new_module



