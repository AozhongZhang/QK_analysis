from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from types import MethodType
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
# from sample_test imporsample
# import torch
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = model.eval()
model = model.cuda()

collected_k_outputs = []
collected_v_outputs = []
def k_proj_hook(module, input, output):
    """
    module: The layer that produced this output (k_proj).
    input:  The input to k_proj.
    output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
    """
    # Detach to avoid growing the autograd graph
    collected_k_outputs.append(output.detach().cpu())

def v_proj_hook(module, input, output):
    """
    Same logic as k_proj_hook, but for v_proj.
    """
    collected_v_outputs.append(output.detach().cpu())

num_layers = len(model.model.layers)
hooks_k = []
hooks_v = []
for layer_idx in range(num_layers):
    # Access the i-th layer
    layer = model.model.layers[layer_idx].self_attn
    print(f"  - K/V heads: {layer.config.num_key_value_heads}")
    
    
    # Register forward hooks
    hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
    hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
    
    hooks_k.append(hook_k)
    hooks_v.append(hook_v)

# print(A)

prompt = "As the Nameless officially do not exist, the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war. While at times this works to their advantage, such as a successful incursion into Imperial territory, other orders cause certain members of the 422nd great distress. One such member, Gusurg, becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven, attached to the ideal of Darcsen independence proposed by their leader, Dahau. At the same time, elements within Gallian Army Command move to erase the Nameless in order to protect their own interests. Hounded by both allies and enemies, and combined with the presence of a traitor within their ranks, the 422nd desperately move to keep themselves alive while at the same time fight to help the Gallian war effort. This continues until the Nameless's commanding officer, Ramsey Crowe, who had been kept under house arrest, is escorted to the capital city of Randgriz in order to present evidence exonerating the weary soldiers and expose the real traitor, the Gallian General that had accused Kurt of Treason."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)


for hook in hooks_k:
    hook.remove()
for hook in hooks_v:
    hook.remove()

print("Num samles (layers) collected:", len(collected_k_outputs))
V = []
K = []
for i in range(len(collected_k_outputs)):
    bsz, seq_len, hidden = collected_k_outputs[i].shape
    v = collected_v_outputs[i].view(bsz, seq_len, 8, (hidden//8)).transpose(1, 2)
    k = collected_k_outputs[i].view(bsz, seq_len, 8, (hidden//8)).transpose(1, 2)
    K.append(k.squeeze(0))
    V.append(v.squeeze(0))
print(K[0].shape)
# print(A)
def plot_pca_heads(matrix: torch.Tensor, target_layer):
    # 假设 all_q: [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = matrix.shape
    print(num_heads)
    print(seq_len)
    print(head_dim)

    # reshape 为 [num_heads * seq_len, head_dim]
    matrix_reshaped = matrix.reshape(num_heads * seq_len, head_dim).cpu().numpy()

    # 为每个向量记录它来自哪个 head
    head_ids = np.repeat(np.arange(num_heads), seq_len)

    pca = PCA(n_components=2)
    q_pca = pca.fit_transform(matrix_reshaped)  # shape: [num_heads * seq_len, 2]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 使用离散 colormap
    cmap = cm.get_cmap("tab20", num_heads)
    scatter = ax.scatter(
        q_pca[:, 0], q_pca[:, 1],
        c=head_ids,
        cmap=cmap,
        s=10,
        alpha=0.7
    )

    # 添加 colorbar（用于标注 Head ID）
    cbar = plt.colorbar(scatter, ticks=np.arange(num_heads))
    cbar.set_label("Head ID")

    # 图示设置
    ax.set_title("PCA of K vectors across all heads")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()      
    plt.savefig(f"/home/azzhang/streaming-llm/head_cluster_llama3/layer{target_layer}_v.png")

# i = 1
for i in range(32):
    plot_pca_heads(V[i].float(), i)




# # import torch
# import torch.nn.functional as F

# def linear_cka_centered_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
#     """
#     A *centered* linear CKA, as in Kornblith et al. (2019), for (L, D) Tensors.
#     This subtracts each row's mean from kv1, kv2 before computing the norm-based formula.
    
#     Steps:
#       1. Row-center each representation (i.e., subtract column means).
#       2. Compute Frobenius norms of X^T X, Y^T Y, X^T Y on the centered data.
#       3. Return (||X^T Y||_F^2) / (||X^T X||_F * ||Y^T Y||_F).

#     Note:
#       - 'Row-center' means we subtract the *column* mean for each dimension (the usual approach 
#         in CKA references). This ensures the average vector over all tokens is zero.

#     Args:
#       kv1: shape (L, D)
#       kv2: shape (L, D)

#     Returns:
#       cka_value: a scalar torch.Tensor
#     """
#     assert kv1.shape[1] == kv2.shape[1], "kv1, kv2 must have same embedding dimension."

#     # Move to GPU if desired
#     device = kv1.device
#     kv1 = kv1.to(device)
#     kv2 = kv2.to(device)
    
#     # 1. Row-center each representation. 
#     #    (Compute column means & subtract => each dimension has mean 0 across L)
#     kv1_centered = kv1 - kv1.mean(dim=0, keepdim=True)
#     kv2_centered = kv2 - kv2.mean(dim=0, keepdim=True)
    
#     # 2. Norm computations
#     xtx = (kv1_centered.T @ kv1_centered).norm(p='fro')
#     yty = (kv2_centered.T @ kv2_centered).norm(p='fro')
#     xty = (kv1_centered.T @ kv2_centered).norm(p='fro')

#     # Handle degenerate case
#     if xtx == 0 or yty == 0:
#         return torch.tensor(0.0, device=device, dtype=kv1.dtype)

#     # 3. Linear CKA formula
#     cka_value = (xty ** 2) / (xtx * yty)

#     return cka_value


# def principal_angle_subspace_similarity_torch(
#     kv1: torch.Tensor, 
#     kv2: torch.Tensor, 
#     rank: int = None
# ) -> torch.Tensor:
#     """
#     Compute a subspace similarity between two matrices kv1, kv2 by comparing 
#     their top-r principal components (SVD subspace).

#     Steps:
#       1. SVD each matrix kv1, kv2 => U1, S1, V1 and U2, S2, V2
#       2. Truncate U1, U2 to top-`rank` columns (top singular vectors)
#       3. Multiply M = U1_r^T @ U2_r
#       4. SVD(M) => get singular values sigma_i
#       5. Return average of (sigma_i^2) => a subspace overlap in [0,1]

#     Args:
#       kv1: shape (L, D)
#       kv2: shape (L, D)
#       rank: how many singular vectors to keep (top-r). If None, 
#             we use min(L, D) for each input to get the full basis.

#     Returns:
#       subspace_sim: scalar tensor in [0,1], the average of sigma^2 over i.
#                     1 => identical subspaces at that rank
#     """
#     # Determine default rank
#     if rank is None:
#         rank = min(kv1.shape[0], kv1.shape[1], kv2.shape[0], kv2.shape[1])

#     # SVD (full_matrices=False) ensures shapes are (L, D) => (L, min(L,D))
#     U1, S1, V1 = torch.svd_lowrank(kv1, q=rank)
#     U2, S2, V2 = torch.svd_lowrank(kv2, q=rank)

#     # Truncate
#     U1_r = U1[:, :rank]  # shape (L, rank)
#     U2_r = U2[:, :rank]  # shape (L, rank)

#     # Multiply => shape (rank, rank)
#     M = U1_r.transpose(0, 1) @ U2_r

#     # SVD on M
#     # The singular values of M measure how well subspaces overlap
#     _, sigma, _ = torch.linalg.svd(M, full_matrices=False)

#     # We return the average of sigma^2
#     subspace_sim = (sigma**2).sum() / rank
#     return subspace_sim
  
  
# def average_cosine_similarity_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
#     """
#     Compute the average rowwise cosine similarity between kv1 and kv2.

#     Specifically, for i in [0..L-1], we do:
#       cosSim( kv1[i], kv2[i] ) =  (kv1[i] dot kv2[i]) / (||kv1[i]|| * ||kv2[i]||)
#     and then average these values over the batch dimension L.

#     Args:
#       kv1: (L, D) - e.g. L rows of D-dimensional embeddings
#       kv2: (L, D) - same shape

#     Returns:
#       A scalar torch.Tensor representing the mean cosine similarity in [-1, 1].
#     """
#     assert kv1.shape == kv2.shape, "kv1 and kv2 must have the same shape"

#     # If needed, move to same device (assuming kv1, kv2 are on same device)
#     # you can also do kv1 = kv1.to(device), kv2 = kv2.to(device) if needed

#     # F.cosine_similarity returns a tensor of shape (L,),
#     # each element is the cosine similarity of row i in kv1 and row i in kv2.
#     cos_sims = F.cosine_similarity(kv1, kv2, dim=0)

#     # Take the average
#     avg_cosine_sim = cos_sims.mean()
#     return avg_cosine_sim


# # %%
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_tensor(w, ax=None, title=None, x_label="Dims", y_label="Token"):
#     # Prepare the data for plotting
#     x = np.arange(w.shape[0])  # Dims
#     y = np.arange(w.shape[1])  # Seqlen
#     x, y = np.meshgrid(x, y)

#     # Use the provided axis or create a new figure and axis
#     if ax is None:
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(111, projection='3d')
#     else:
#         fig = None

#     # Plot the surface
#     surf = ax.plot_surface(x, y, w.T, cmap='coolwarm')
#     ax.xaxis.set_tick_params(pad=-6, size=13)
#     ax.yaxis.set_tick_params(pad=-4, size=13)
#     ax.zaxis.set_tick_params(pad=-3, size=13)
    
#     # Set title and labels
#     if title is not None:
#         ax.set_title(title, fontsize=15)
#     ax.set_ylabel(x_label, labelpad=-5, fontsize=14)
#     ax.set_xlabel(y_label, labelpad=-1, fontsize=14)
#     ax.set_zlabel('Absolute Activation Value', labelpad=-5, fontsize=14)

#     return fig

# def plot_vector(w, ax=None, title=None, x_label="Dims", y_label="Value"):
#     # Use the provided axis or create a new figure and axis
#     if ax is None:
#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(111)
#     else:
#         fig = None

#     # Plot the vector
#     ax.plot(w)
#     ax.xaxis.set_tick_params(size=16)
#     ax.yaxis.set_tick_params(size=16)
#     if x_label is not None:
#         ax.set_xlabel(x_label, fontsize=14)
#     if y_label is not None:
#         ax.set_ylabel(y_label, fontsize=14)
#     # Set title
#     if title is not None:
#         ax.set_title(title)

#     return fig


# def plot_side_by_side(tensors, titles=None, plot_type='tensor', saved_path = None):
#     """
#     Plot multiple tensors or vectors side by side.

#     Args:
#         tensors: List of tensors or vectors to plot.
#         titles: List of titles for each plot (optional).
#         plot_type: Type of plot ('tensor' for 3D tensor, 'vector' for 1D vector).
#     """
#     num_plots = len(tensors)
#     fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots + 3, 6), subplot_kw={'projection': '3d'} if plot_type == 'tensor' else {})

#     # In case of only one plot, axs won't be an array; make it an array for consistency
#     if num_plots == 1:
#         axs = [axs]

#     # Plot each tensor/vector
#     for i, (tensor, ax) in enumerate(zip(tensors, axs)):
#         title = titles[i] if titles is not None else None
#         if plot_type == 'tensor':
#             plot_tensor(tensor, ax=ax, title=title)
#         elif plot_type == 'vector':
#             plot_vector(tensor, ax=ax, title=title)
#         else:
#             raise ValueError("Unsupported plot type. Use 'tensor' or 'vector'.")

#     plt.tight_layout()
#     if saved_path is not None:
#         plt.savefig(saved_path)
#     plt.show()

# # %%
# import torch
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
# import seaborn as sns

# def plot_heatmap(tensor, title="Heatmap", custom_colors=None, colorbar=True, 
#                  x_label="", y_label="", font_size=14, tick_font_size=12, interpolation="bilinear"):
#     """
#     Plots a smooth heatmap from a 2D PyTorch tensor with a custom color gradient.

#     Args:
#         tensor (torch.Tensor): 2D tensor to plot.
#         title (str): Title of the heatmap.
#         custom_colors (list): List of 6 colors for the colormap.
#         colorbar (bool): Whether to show the colorbar.
#         x_label (str): Label for the x-axis.
#         y_label (str): Label for the y-axis.
#         font_size (int): Font size for title and labels.
#         tick_font_size (int): Font size for x and y ticks.
#         interpolation (str): Interpolation method for smooth transitions (e.g., 'bilinear', 'bicubic').
#     """
#     if not isinstance(tensor, torch.Tensor):
#         raise TypeError("Input must be a PyTorch tensor.")
    
#     if tensor.dim() != 2:
#         raise ValueError("Input tensor must be 2D.")
    
#     # Convert tensor to numpy
#     matrix = tensor.detach().cpu().numpy()

#     # Define a custom colormap
#     # if custom_colors and len(custom_colors) >= 2:
#     #     cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
#     # else:
#     #     cmap = "coolwarm"  # Default colormap if none provided

#     cmap = sns.diverging_palette(240, 10, as_cmap=True)
#     #cmap = sns.color_palette("vlag", as_cmap=True)
#     #cmap = 'coolwarm'
#     # Plot heatmap with smooth transitions
#     plt.figure(figsize=(7, 6))
#     im = plt.imshow(matrix, cmap=cmap, aspect='auto')

#     # Set title and labels
#     plt.title(title, fontsize=23)
#     plt.xlabel(x_label, fontsize=21)
#     plt.ylabel(y_label, fontsize=21)
    
#     # Configure tick parameters
#     plt.xticks(fontsize=19)
#     plt.yticks(fontsize=19)

#     # Configure colorbar
#     if colorbar:
#         cbar = plt.colorbar(im)
#         cbar.ax.tick_params(labelsize=19)
#     plt.savefig(f"{title}.pdf", bbox_inches='tight')
#     plt.show()

# # %% [markdown]
# # ## CKA Analysis

# # %%
# # cka_matrix = torch.zeros(num_layers, num_layers)
# cka_matrix = torch.zeros(32, 32)
# mode = "Value"
# # mode = "Key"
# if mode == "Key":

#     target_layer = 31
#     num_heads, seq_len, head_dim = K[target_layer].shape
#     for i in range(num_heads):
#         for j in range(num_heads):
#             vi = K[target_layer][i]
#             vj = K[target_layer][j]
#             assert vi.shape == vj.shape
#             vi = vi.cuda().float()
#             vj = vj.cuda().float()
#             print(vi.shape)
#             cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
#             print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")

#     # for i in range(num_layers):
#     #     for j in range(num_layers):
#     #         ki = collected_k_outputs[i]
#     #         kj = collected_k_outputs[j]
            
#     #         assert ki.shape == kj.shape
#     #         assert ki.shape[0] == 1 # batch size is 1
            
#     #         ki = ki.squeeze(0).cuda().float()
#     #         kj = kj.squeeze(0).cuda().float()
#     #         cka_matrix[i, j] = linear_cka_centered_torch(ki, kj)
#     #         print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")
            
#             # del ki, kj
# elif mode == "Value":
#     # print(collected_v_outputs.shape)
#     # V_0 = collected_v_outputs[0]
#     # print(V_0.shape)
#     # print(A)
#     target_layer = 10
#     num_heads, seq_len, head_dim = V[target_layer].shape
#     for i in range(num_heads):
#         for j in range(num_heads):
#             vi = V[target_layer][i]
#             vj = V[target_layer][j]
#             assert vi.shape == vj.shape
#             vi = vi.cuda().float()
#             vj = vj.cuda().float()
#             print(vi.shape)
#             cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
#             print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")

#     # for i in range(num_layers):
#     #     for j in range(num_layers):
#     #         vi = collected_v_outputs[i]
#     #         vj = collected_v_outputs[j]
            
#     #         assert vi.shape == vj.shape
#     #         assert vi.shape[0] == 1

#     #         vi = vi.squeeze(0).cuda().float()
#     #         vj = vj.squeeze(0).cuda().float()
#     #         cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
#     #         print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")
# elif mode == "KV":
#     for i in range(num_layers):
#         for j in range(num_layers):
#             ki = collected_k_outputs[i]
#             vj = collected_v_outputs[j]
            
#             assert not torch.all(ki == vj)
            
#             assert ki.shape == vj.shape
#             assert ki.shape[0] == 1
            
#             ki = ki.squeeze(0).cuda().float()
#             vj = vj.squeeze(0).cuda().float()
#             cka_matrix[i, j] = linear_cka_centered_torch(ki, vj)
#             print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")

# # %%
# cmap = sns.diverging_palette(240, 10)

# # %%
# custom_colors = ["#D93F49", "#E28187", "#EBBFC2", "#D5E1E3", "#AFC9CF", "#8FB4BE"]  
# custom_colors = custom_colors[::-1]  # Reverse the colors for better contrast
# plot_heatmap(cka_matrix, title=f"CKA Matrix ({mode}-Cache)", colorbar=True, x_label="Layer", y_label="Layer", custom_colors=custom_colors)


