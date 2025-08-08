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

def projection(weight_head):
    original_dtype = weight_head.dtype
    X = weight_head - weight_head.mean(dim=1, keepdim=True)

    # 2. compute cov matrix based on column
    cov = X @ X.T / X.shape[1]  # shape: [head_dim, head_dim]

    # 3. 做 eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov.to(torch.float32))  # eigvecs: [head_dim, head_dim]

    # 4. construct whitening projection
    eps = 1e-6
    D_inv_sqrt = torch.diag(1.0 / (eigvals + eps).sqrt())  # [head_dim, head_dim]
    whiten_proj = eigvecs @ D_inv_sqrt @ eigvecs.T  # shape: [head_dim, head_dim]

    return whiten_proj.to(original_dtype)

# import torch
import torch.nn.functional as F

def hadamard_projection(weight_head: torch.Tensor) -> torch.Tensor:
    """
    使用 Hadamard 矩阵对 weight_head 进行投影。
    输入 shape: [head_dim, in_feature]
    返回 shape: [head_dim, in_feature]
    """
    head_dim = weight_head.shape[0]

    # 找到下一个最近的 2 的幂次
    def next_power_of_two(n):
        return 1 << (n - 1).bit_length()

    target_dim = next_power_of_two(head_dim)
    pad = target_dim - head_dim

    # 构造 Hadamard 矩阵
    def generate_hadamard(n):
        assert (n & (n - 1) == 0), "Hadamard size must be power of 2"
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0)
        return H / (n ** 0.5)  # 通常归一化保持单位正交

    H = generate_hadamard(target_dim).to(weight_head.device, weight_head.dtype)

    # padding weight if needed
    # if pad > 0:
    #     weight_head = F.pad(weight_head, (0, 0, 0, pad))  # pad along head_dim

    # 投影
    # projected = H @ weight_head  # shape: [target_dim, in_feature]

    # # 如果 padding 了，剪掉
    # if pad > 0:
    #     projected = projected[:head_dim]

    return H


class HeadwiseLowRankModule(nn.Module):
    """ Headwise low rank module """

    def __init__(self, ranks, in_features, out_features, bias):
    # def __init__(self, ranks, inverse_indices, group_to_rows, original_layer, in_features, out_features, bias):
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
    def from_linear_whiten_double_svd(
        old_module: nn.Linear, # [out_features, in_features] = [num_head*head_dim, in_features]
        ranks: list,
        KV_for_first_svd,
    ):   
        
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        from matplotlib import pyplot as plt
        num_head, seq_len, head_dim = KV_for_first_svd.shape
        # print(num_head)
        print(seq_len)
        # print(head_dim)
        KV = KV_for_first_svd.reshape(seq_len, -1)
        U, S, Vt = torch.linalg.svd(KV.float(), full_matrices=False)
        print(U.shape)
        U_heads = U.reshape(seq_len, num_head, -1).transpose(0, 1)
        energy_per_head = (U_heads ** 2).sum(dim=1)
        var_across_heads = energy_per_head.var(dim=0) 
        var_cpu = var_across_heads.detach().cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.plot(var_cpu, marker='o')
        plt.title("Variance per SVD component across heads (GPU SVD)")
        plt.xlabel("SVD Component Index")
        plt.ylabel("Variance across heads")
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"/home/azzhang/Palu/img/single_svd_{ranks[0]}.png")
        # plt.show()

        # plt.figure()
        # _, s, _ = torch.linalg.svd(old_module.weight.data.float())
        
        # plt.plot(s.detach().cpu().numpy(), label="Straight SVD")
        
        w_ = old_module.weight.data.reshape(32, -1, old_module.in_features) # [num_head, head_dim, in_feature] = [32, 128, 4096]
        num_head, head_dim, in_features = w_.shape
        # print("original weight: ", w_)
        Vt_first = []
        for i in range(num_head):
            w_head = w_[i] # [head_dim, in_features]
            kv_head = KV_for_first_svd[i]
            # print(kv_head.shape)
            v_head = _per_head_decomposition_from_weight_first_svd(kv_head.to(w_.device)) # [head_dim_R, head_dim]
            # print(v_head.shape)
            Vt_first.append(v_head)
            w_[i] = (w_head.T @ v_head.T).T # [head_dim_R, in_features]
        
        # print("projected weight: ", w_)
        w = w_.reshape(-1, old_module.in_features)

        # _, s1, _ = torch.linalg.svd(w.float())
        # plt.plot(s1.detach().cpu().numpy(), label="2-step SVD")
        # plt.legend()
        # plt.savefig(f"/home/azzhang/Palu/img/realcase_{ranks[0]}_origianl.png")
        w = w.reshape(len(ranks), -1, old_module.in_features) # [num_group, group_dim (concat head_dim_R), in_featrue]
        

        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):

            # construct group Vt_first
            V_block_diag_group = torch.zeros((num_head * head_dim)//len(ranks), (num_head * head_dim)//len(ranks), device=old_module.weight.device)
            for j in range(num_head//len(ranks)):
                idx = i * (num_head//len(ranks)) + j
                start = j * head_dim
                V_block_diag_group[start:start+head_dim, start:start+head_dim] = Vt_first[idx].T
            
            # w[i]= group_dim (concat head_dim_R), in_featrue
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (group_dim (concat head_dim_R), rank), r: (rank, in_featrue)
            wl.append(V_block_diag_group.half() @ l)
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
    def from_linear_whiten_H_projection(
        old_module: nn.Linear, # [out_features, in_features] = [num_head*head_dim, in_features]
        ranks: list,
        # KV_for_first_svd,
    ):   
        
        new_module = HeadwiseLowRankModule(ranks, old_module.in_features, old_module.out_features, bias=old_module.bias is not None)
        
        w_ = old_module.weight.data.reshape(32, -1, old_module.in_features) # [num_head, head_dim, in_feature] = [32, 128, 4096]
        num_head, head_dim, in_features = w_.shape
        # print("original weight: ", w_)
        Vt_first = []
        for i in range(num_head):
            from matplotlib import pyplot as plt
            plt.figure()
            w_head = w_[i] # [head_dim, in_features]
            _, s, _ = torch.linalg.svd(w_head.float())
            plt.plot(s.detach().cpu().numpy(), label='original')
            # whiten_projection = projection(w_head)
            hada_projection = hadamard_projection(w_head)
            
            Vt_first.append(hada_projection)
            # w_[i] = (w_head.T @ whiten_projection.T).T # [head_dim_R, in_features]
            w_[i] = (w_head.T @ hada_projection.T).T

            _, s1, _ = torch.linalg.svd((w_head.T @ hada_projection.T).T.float())
            plt.plot(s1.detach().cpu().numpy(), label='projected')
            plt.legend()
            plt.savefig(f"/home/azzhang/Palu/img_svd/com_hada.png")
            print(A)
        
        # print("projected weight: ", w_)
        w = w_.reshape(-1, old_module.in_features)

        w = w.reshape(len(ranks), -1, old_module.in_features) # [num_group, group_dim (concat head_dim_R), in_featrue]
        
        # Handle the cases where the bias is not None
        if old_module.bias is not None:
            b = old_module.bias.data.reshape(len(ranks), -1)
        
        wl = []
        wr = []
        for i in range(len(ranks)):

            # construct group Vt_first
            V_block_diag_group = torch.zeros((num_head * head_dim)//len(ranks), (num_head * head_dim)//len(ranks), device=old_module.weight.device)
            for j in range(num_head//len(ranks)):
                idx = i * (num_head//len(ranks)) + j
                start = j * head_dim
                V_block_diag_group[start:start+head_dim, start:start+head_dim] = Vt_first[idx].T
            
            # w[i]= group_dim (concat head_dim_R), in_featrue
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            # l: (group_dim (concat head_dim_R), rank), r: (rank, in_featrue)
            wl.append(V_block_diag_group.half() @ l)
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
            from matplotlib import pyplot as plt
            plt.figure()
            _, s, _ = torch.linalg.svd(w[i].float())
            plt.plot(s.detach().cpu().numpy(), label='original')
            l, r = _per_head_whiten_decomposition_from_weight(w[i], old_module.scaling_diag_matrix, ranks[i])
            plt.legend()
            plt.savefig(f"/home/azzhang/Palu/img_svd/groupsvd.png")
            print(A)
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


