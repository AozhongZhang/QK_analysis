import torch
import torch.nn as nn
from .quant import Quantizer
from .hadamard_utils import apply_hadamard

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
    
    U_H, Sigma_H, V_H = torch.svd(H)
    Sigma_H_sqrt = torch.sqrt(Sigma_H) 
    X_0 = Sigma_H_sqrt.unsqueeze(1) * V_H.T
    
    X_0M = X_0 @ M
    U, Sigma, V = torch.svd(X_0M)
    
    U_r = U[:, :r]
    Sigma_r = torch.diag(Sigma[:r])
    V_r = V[:, :r]
    
    A = V_H @ torch.diag(1.0 / Sigma_H_sqrt) @ (U_r @ Sigma_r)
    B = V_r.T
    
    return B.T, A.T
    # return A, B


class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    # def __init__(self, ranks, inverse_indices, group_to_rows, in_features, out_features, bias):
    def __init__(self, ranks, inverse_indices, group_to_rows, in_features, out_features, bias):
        super().__init__()
        self.ranks = ranks
        # changes
        
        self.inverse_indices = inverse_indices
        self.group_to_rows = group_to_rows
        self.num_groups = len(ranks)
        self.in_features = in_features
        self.out_features = out_features
        # self.group_dim = out_features // self.num_groups
        
        group_sizes = [len(rows) for _, rows in sorted(group_to_rows.items())]
        total_heads = sum(group_sizes)
        group_dims = [out_features * size / total_heads for size in group_sizes]
        self.group_dims = group_dims
        # print(group_dims)

        # if (self.group_dim * self.num_groups) != self.out_features:
        #     raise ValueError(
        #         f"out_features must be divisible by num_groups (got `out_features`: {self.out_features}"
        #         f" and `num_groups`: {self.num_groups})."
        #     )

        self.VT = nn.Linear(in_features, sum(ranks), bias=False)
        # self.U = nn.ModuleList([
        #     nn.Linear(r, self.group_dim, bias=bias) for r in ranks
        # ])
        self.U = nn.ModuleList([
            nn.Linear(int(r), int(gdim), bias=bias)
            for r, gdim in zip(ranks, group_dims)
        ])

        # self.original_layer = original_layer
        self.quantized_latents = False
        self.latent_quantizer = None
        self.last_mse = None

    def forward(self, hidden_states: torch.Tensor):
        latents = self.project_to_latent(hidden_states)
        if self.quantized_latents:
            latents = self.quantize_latent(latents)

        output = self.reconstruct_permuted(latents)

        # # compute MSE with reference linear layer if provided
        # if self.original_layer is not None:
        #     import torch.nn.functional as F
        #     with torch.no_grad():
        #         target = self.original_layer(hidden_states)
        #         self.last_mse = F.mse_loss(output, target).item()
        #         print(self.original_layer)
        #         print(self.last_mse)
        #         print("+++++++++++++++++++")
        
        # return self.reconstruct_permuted(latents)
        return output

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
    
    def reconstruct_permuted(self, latents: torch.Tensor):
        outputs = []
        offset = 0
        for i, r in enumerate(self.ranks):
            out = self.U[i](latents[:, :, offset:offset + r])
            outputs.append(out)
            offset += r
        output = torch.cat(outputs, dim=-1)
        
        # [B, L, H * D]
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
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
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
    def from_linear(
        old_module: nn.Linear,
        ranks: list,
    ):
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
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
    def from_linear_exact(
        old_module: nn.Linear,
        ranks: list,
        # hessian: torch.Tensor,  # shape: [in_features, in_features]
        ):
        """
        Convert a Linear module to HeadwiseLowRankModule using exact_solution.

        Args:
            old_module: nn.Linear, weight shape [out_features, in_features]
            ranks: list of int, per-head low-rank values
            hessian: torch.Tensor, approximated input covariance (H = X^T X)
                 shape = [in_features, in_features]
        """
        new_module = HeadwiseLowRankModule(
            ranks=ranks,
            in_features=old_module.in_features,
            out_features=old_module.out_features,
            bias=old_module.bias is not None
        )

        # Reshape weights into per-group heads
        w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group, out_featrue//num_group, in_feature]
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)

        wl = []
        wr = []

        for i in range(len(ranks)):
            group_weight = w[i]  # shape: [group_dim, in_features] [512, 4096] if num_group=8, then group_dim=512
            r = ranks[i]
            # Use shared hessian for all groups (if per-group not available)
            l, r_ = _per_head_decomposition_from_weight_calibration(old_module.hessian_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
            wl.append(l)
            wr.append(r_)

        # Load U
        for i in range(len(ranks)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].to(new_module.U[i].weight.dtype).contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # Load VT
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight.to(new_module.VT.weight.dtype)

        return new_module
    @staticmethod
    def from_linear_exact_new(
        old_module: nn.Linear,
        ranks: list,
        group_to_rows: dict,
        ):
        """
        Convert a Linear module to HeadwiseLowRankModule using exact_solution.

        Args:
            old_module: nn.Linear, weight shape [out_features, in_features]
            ranks: list of int, per-head low-rank values
            hessian: torch.Tensor, approximated input covariance (H = X^T X)
                 shape = [in_features, in_features]
        """
                
        permuted_indices = []
        for g in sorted(group_to_rows.keys()):
            permuted_indices.extend(group_to_rows[g])

        inverse_indices = [0] * len(permuted_indices)
        for new_idx, original_idx in enumerate(permuted_indices):
            inverse_indices[original_idx] = new_idx
        # print(inverse_indices)
        new_module = HeadwiseLowRankModule(
            ranks,
            inverse_indices,
            group_to_rows=group_to_rows,
            in_features=old_module.in_features,
            out_features=old_module.out_features,
            bias=old_module.bias is not None,
        )

        # Reshape weights into per-group heads
        total_num_rows = sum(len(v) for v in group_to_rows.values())
        
        w = old_module.weight.data.reshape(total_num_rows, -1, old_module.in_features) # [num_group, group_dim, in_features]
        w = w[permuted_indices] # [num_group, group_dim, in_features] permuted group index
        w = w.reshape(-1, old_module.in_features) # [out_features, in_features] permuted
        head_dim = w.shape[0] // total_num_rows
        group_sizes = [len(rows) for rows in group_to_rows.values()]
        split_sizes = [n * head_dim for n in group_sizes]
        group_weights = torch.split(w, split_sizes, dim=0)
        group_weights = list(group_weights)

        if old_module.bias is not None:
            b = old_module.bias.data.reshape(total_num_rows, -1)
            b = b[permuted_indices].reshape(-1)

        


        # w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group, out_featrue//num_group, in_feature]
        # if old_module.bias is not None:
        #     b = old_module.bias.data.reshape(len(ranks), -1)

        wl = []
        wr = []

        for i in range(len(group_weights)):
            group_weight = group_weights[i]  # shape: [group_dim, in_features] [512, 4096] if num_group=8, then group_dim=512
            r = ranks[i]
            # Use shared hessian for all groups (if per-group not available)
            # l, r_ = _per_head_decomposition_from_weight_calibration(old_module.scaling_diag_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
            l, r_ = _per_head_decomposition_from_weight_calibration(old_module.hessian_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
            # l, r_ = _per_head_whiten_decomposition_from_weight(group_weight, old_module.scaling_diag_matrix, ranks[i])
            wl.append(l)
            wr.append(r_)

        # for i in range(len(ranks)):
        #     group_weight = w[i]  # shape: [group_dim, in_features] [512, 4096] if num_group=8, then group_dim=512
        #     r = ranks[i]
        #     # Use shared hessian for all groups (if per-group not available)
        #     l, r_ = _per_head_decomposition_from_weight_calibration(old_module.hessian_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
        #     wl.append(l)
        #     wr.append(r_)

        # Load U
        # for i in range(len(ranks)):
        #     if new_module.U[i].weight.data.shape != wl[i].shape:
        #         raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
        #     new_module.U[i].weight.data = wl[i].to(new_module.U[i].weight.dtype).contiguous()
        #     if old_module.bias is not None:
        #         new_module.U[i].bias.data = b[i]
        
        for i in range(len(group_weights)):
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].to(new_module.U[i].weight.dtype).contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # Load VT
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight.to(new_module.VT.weight.dtype)

        return new_module
    
    @staticmethod
    def from_linear_whiten_new(
        old_module: nn.Linear,
        ranks: list,
        group_to_rows: dict,
        # permuted_indices: list,
        # inverse_indices: list,
        # hessian: torch.Tensor,  # shape: [in_features, in_features]
        ):
        """
        Convert a Linear module to HeadwiseLowRankModule using exact_solution.

        Args:
            old_module: nn.Linear, weight shape [out_features, in_features]
            ranks: list of int, per-head low-rank values
            hessian: torch.Tensor, approximated input covariance (H = X^T X)
                 shape = [in_features, in_features]
        """
                
        permuted_indices = []
        for g in sorted(group_to_rows.keys()):
            permuted_indices.extend(group_to_rows[g])

        inverse_indices = [0] * len(permuted_indices)
        for new_idx, original_idx in enumerate(permuted_indices):
            inverse_indices[original_idx] = new_idx

        new_module = HeadwiseLowRankModule(
            ranks=ranks,
            # permuted_indices,
            inverse_indices=inverse_indices,
            group_to_rows=group_to_rows,
            original_layer=old_module,
            in_features=old_module.in_features,
            out_features=old_module.out_features,
            bias=old_module.bias is not None
        )

        # Reshape weights into per-group heads
        total_num_rows = sum(len(v) for v in group_to_rows.values())
        # total_num_rows = len(permuted_indices)
        w = old_module.weight.data.reshape(total_num_rows, -1, old_module.in_features)
        w = w[permuted_indices]
        w = w.reshape(-1, old_module.in_features)
        head_dim = w.shape[0] // total_num_rows
        group_sizes = [len(rows) for rows in group_to_rows.values()]
        split_sizes = [n * head_dim for n in group_sizes]
        group_weights = torch.split(w, split_sizes, dim=0)
        group_weights = list(group_weights)

        if old_module.bias is not None:
            b = old_module.bias.data.reshape(total_num_rows, -1)
            b = b[permuted_indices].reshape(-1)

        


        # w = old_module.weight.data.reshape(len(ranks), -1, old_module.in_features) # [num_group, out_featrue//num_group, in_feature]
        # if old_module.bias is not None:
        #     b = old_module.bias.data.reshape(len(ranks), -1)

        wl = []
        wr = []

        for i in range(len(group_weights)):
            group_weight = group_weights[i]  # shape: [group_dim, in_features] [512, 4096] if num_group=8, then group_dim=512
            r = ranks[i]
            # Use shared hessian for all groups (if per-group not available)
            # l, r_ = _per_head_decomposition_from_weight_calibration(old_module.hessian_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
            l, r_ = _per_head_whiten_decomposition_from_weight(group_weight, old_module.scaling_diag_matrix, ranks[i])
            wl.append(l)
            wr.append(r_)

        # for i in range(len(ranks)):
        #     group_weight = w[i]  # shape: [group_dim, in_features] [512, 4096] if num_group=8, then group_dim=512
        #     r = ranks[i]
        #     # Use shared hessian for all groups (if per-group not available)
        #     l, r_ = _per_head_decomposition_from_weight_calibration(old_module.hessian_matrix.to(group_weight.device), group_weight.T.to(torch.float32), r)
        #     wl.append(l)
        #     wr.append(r_)

        # Load U
        # for i in range(len(ranks)):
        #     if new_module.U[i].weight.data.shape != wl[i].shape:
        #         raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
        #     new_module.U[i].weight.data = wl[i].to(new_module.U[i].weight.dtype).contiguous()
        #     if old_module.bias is not None:
        #         new_module.U[i].bias.data = b[i]
        
        for i in range(len(group_weights)):
            # print(new_module.U[i].weight.data.shape)
            # print(wl[i].shape)
            # print("++++")
            if new_module.U[i].weight.data.shape != wl[i].shape:
                raise ValueError(f"{new_module.U[i].weight.data.shape} != {wl[i].shape}")
            new_module.U[i].weight.data = wl[i].to(new_module.U[i].weight.dtype).contiguous()
            if old_module.bias is not None:
                new_module.U[i].bias.data = b[i]

        # Load VT
        VT_weight = torch.cat(wr, dim=0).contiguous()
        assert new_module.VT.weight.data.shape == VT_weight.shape
        new_module.VT.weight.data = VT_weight.to(new_module.VT.weight.dtype)

        return new_module


