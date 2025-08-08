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

def _per_head_decomposition_from_weight_first_svd(weight):

    original_dtype = weight.dtype
    # Get weight matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    return Vt.to(original_dtype)

class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(self, ranks, in_features, out_features, bias):
        super().__init__()
        self.ranks = ranks
        # changes
        
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
        self.U = nn.ModuleList([
            nn.Linear(r, self.group_dim, bias=bias) for r in ranks
        ])

        self.quantized_latents = False
        self.latent_quantizer = None
        self.last_mse = None

    def forward(self, hidden_states: torch.Tensor):
        latents = self.project_to_latent(hidden_states)
        if self.quantized_latents:
            latents = self.quantize_latent(latents)

        output = self.reconstruct(latents)

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
    def from_linear_whiten_double_svd(
        old_module: nn.Linear, # [out_features, in_features] = [num_head*head_dim, in_features]
        ranks: list,
        KV_for_first_svd, # the K/V matrix for the first svd [num_head, seq_len, head_dim]
    ):   
        
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        
        w_ = old_module.weight.data.reshape(32, -1, old_module.in_features) # [num_head, head_dim, in_feature] = [32, 128, 4096]
        num_head, head_dim, in_features = w_.shape
        
        Vt_first = []
        for i in range(num_head):
            w_head = w_[i] # [head_dim, in_features]
            kv_head = KV_for_first_svd[i]
            v_head = _per_head_decomposition_from_weight_first_svd(kv_head.to(w_.device)) # [head_dim_R, head_dim]
            Vt_first.append(v_head)
            w_[i] = (w_head.T @ v_head.T).T # projection [head_dim_R, in_features]
        
        w = w_.reshape(-1, old_module.in_features)
        w = w.reshape(len(ranks), -1, old_module.in_features) # [num_group, group_dim (concat head_dim_R), in_featrue]
        
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):

            # construct group block diag Vt_first
            V_block_diag_group = torch.zeros((num_head * head_dim)//len(ranks), (num_head * head_dim)//len(ranks), device=old_module.weight.device)
            for j in range(num_head//len(ranks)):
                idx = i * (num_head//len(ranks)) + j
                start = j * head_dim
                V_block_diag_group[start:start+head_dim, start:start+head_dim] = Vt_first[idx].T
            
            # w[i]= group_dim (concat head_dim_R), in_featrue
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (group_dim (concat head_dim_R), rank), r: (rank, in_featrue)
            wl.append(V_block_diag_group.half() @ l) # times back
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


