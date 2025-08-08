import math
import os, click
import torch
import torch.nn as nn
from loguru import logger
# from .model import AVAILABLE_MODELS
from .data_utils import get_calib_data
import math
from tqdm import tqdm
import re, math, torch
from collections import defaultdict
from typing import Dict, List, Tuple

def rounding_search_result(config: dict, block_size=32):
    for module_name in config.keys():
        ranks = config[module_name]
        for i in range(len(ranks)):
            ranks[i] = max(1, round(ranks[i] / block_size)) * block_size
        config[module_name] = ranks
    return config

def split_values(data, group_number):
    result = {}
    for key, value in data.items():
        new_value = [v // group_number for v in value for _ in range(group_number)]
        result[key] = new_value
    return result

def _per_layer_decomposition_from_weight_svd(weight):

    original_dtype = weight.dtype
    # Get K matrix decomposed
    U, S, Vt = torch.linalg.svd(weight.to(torch.float32), full_matrices=False)

    return (S ** 2).sum().to(original_dtype)

# def calib_energy_info(model, calib_loader, device, group_to_rows_all_layers, use_cache=True):
def calib_energy_info(model, device, use_cache=True):
    model.half()
    model.to(device)
    model_id = model.config._name_or_path
    cache_file = f"/home/azzhang/Palu/cache/{model_id.replace('/','_')}_calib_energy_info.pt"

    logger.info(f"[Fisher] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and use_cache:
        logger.info(f"[Fisher] File {cache_file} exist.", fg="green")
        logger.info(f"[Fisher] Load cache_file={cache_file}", fg="yellow")
        energy_info_all_layers = torch.load(cache_file, map_location="cpu")
        # print(all_fisher_info)
        # for name, module in model.named_modules():
        #     if isinstance(module, nn.Linear) and "attn" in name:
        #         module.energy = all_fisher_info[name].to(module.weight.device)
    else:
        model.eval()
        logger.info(f"[Fisher] No cache_file={cache_file}", fg="red")
        logger.info(f"[Fisher] Create fisher info list...", fg="yellow")

        # ====== group-based fisher score  ======
        energy_info_all_layers = {}

        for name, module in model.named_modules():
            # if isinstance(module, nn.Linear) and "attn" in name:
            if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
                # module.energy = 0
                energy = _per_layer_decomposition_from_weight_svd(module.weight.data)
                energy_info_all_layers[name] = energy.item()


        # save cache
        logger.info(f"[Energy] Save the fisher info list to:  {cache_file}", fg="yellow")
        torch.save(energy_info_all_layers, cache_file)

        # save cache
        # all_energy_info = {}
        # for name, module in model.named_modules():
        #     if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
        #         module._forward_hooks.clear()
        #         all_energy_info[name] = module.fisher_info

        # logger.info(f"[Fisher] Save the fisher info list to:  {cache_file}", fg="yellow")
        # torch.save(all_energy_info, cache_file)

    
    return energy_info_all_layers



def _get_layer_id_from_name(name: str) -> int:
    """
    提取诸如
      model.layers.12.self_attn.k_proj
      transformer.h.31.attn.k_proj
      decoder.block.7.k_proj
    这类名字中的层号。找不到就抛 ValueError。
    """
    m = re.search(r'\.(\d+)\.', name)
    if m is None:
        raise ValueError(f"Cannot parse layer id from {name}")
    return int(m.group(1))

def _detect_head_dim_KV(model, layer_id: int) -> int:
    """
    在给定层里找一个 k_proj / v_proj，推断 head_dim：
      head_dim = out_features // num_heads_in_that_layer
    其中 num_heads_in_layer 由 group_to_rows_all_layers 决定（外层调用传入）。
    """
    for n, mod in model.named_modules():
        try:
            lid = _get_layer_id_from_name(n)
        except ValueError:
            continue
        if lid != layer_id:
            continue
        if ("k_proj" in n or "v_proj" in n) and isinstance(mod, torch.nn.Linear):
            return mod.weight.shape[0]  # 仍需要除以 num_heads，外层再处理
    raise RuntimeError(f"No k_proj/v_proj found for layer {layer_id}")

import torch
from typing import Union

def detect_head_dim_from_name(
    model: torch.nn.Module,
    layer_name: str
) -> int:
    """
    根据完整的 layer_name（如 'model.layers.30.self_attn.v_proj'），
    在模型里找到对应的 nn.Linear 并返回 out_features。

    Args:
        model (torch.nn.Module): 已加载的模型
        layer_name (str)       : 完整模块路径，只支持 k_proj / v_proj

    Returns:
        int: module.weight.shape[0] (即 out_features)，
             外层再用 out_features // num_heads 获得真正 head_dim。
    """
    if not (".k_proj" in layer_name or ".v_proj" in layer_name):
        raise ValueError("layer_name 必须包含 'k_proj' 或 'v_proj'")

    for name, module in model.named_modules():
        if name == layer_name:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{layer_name} 不是 nn.Linear，而是 {type(module)}")
            return module.weight.shape[0]

    raise RuntimeError(f"未找到名为 '{layer_name}' 的模块")


def allocate_rank_from_group_rows_and_energy(
        model,
        tokenizer,
        group_to_rows_all_layers: Dict[str, Dict[int, List[int]]],
        # fisher_info_all_layers: Dict[int, List[float]],
        param_ratio_target: float
) -> Tuple[Dict[int, List[int]], int, int]:
    """
    返回:
        select_result   : {layer_id: [rank_group0, rank_group1, ...]}
        final_rank_sum  : 实际分到的总 rank
        total_max_rank  : 所有 group 最大 rank 之和 (未压缩)
    """


    # calib_loader = get_calib_data("wikitext2", tokenizer, "meta-llama/Llama-2-7b-hf", 2048, seqlen=1024)
    enery_info_all_layers = calib_energy_info(model, model.device, True)
    # --------------- Step 0. 预计算每层 head_dim & group_max_rank ----------------
    max_ranks_all_layers = {}
    for layer_id, group_to_rows in group_to_rows_all_layers.items():
        num_heads_layer = sum(len(r) for r in group_to_rows.values())
        raw_out_features = detect_head_dim_from_name(model, layer_id)          # 还没除 head 数
        # raw_out_features = _detect_head_dim_KV(model, layer_id)          # 还没除 head 数
        head_dim = raw_out_features // num_heads_layer                # 真正 head_dim
        max_ranks_all_layers[layer_id] = [len(r) * head_dim for r in group_to_rows.values()]

    # --------------- Step 1. 统计总 rank 与 energy 总权重 --------------------------
    total_max_rank = sum(sum(x) for x in max_ranks_all_layers.values())
    target_total_rank = int(total_max_rank * param_ratio_target)
    print("max_ranks_all_layers: ", max_ranks_all_layers)
    print("+++++++++++++++++++++++++++++++++")
    print("enery_info_all_layers: ", enery_info_all_layers)
    print("total_max_rank: ", total_max_rank)

    # fisher_sum = sum(sum(lst) for lst in enery_info_all_layers.values())
    # energy_dict = {k: v.item() for k, v in enery_info_all_layers.items()}
    total_energy  = sum(enery_info_all_layers.values())
    print("total_energy: ", total_energy)

    # --------------- Step 2. 先做 floor 分配 -------------------------------------
    rank_float, rank_int = defaultdict(list), defaultdict(list)
    # rank_float, rank_int = {}, {}
    flat_index = []   # (layer_id, gid) 记录 residual 排序

    for layer_id, layer_energy in enery_info_all_layers.items(): # each layer
        
        r_float = target_total_rank * layer_energy / total_energy
        r_float = min(r_float, sum(max_ranks_all_layers[layer_id]))
        rank_float[layer_id].append(r_float)
        rank_int[layer_id].append(int(r_float))
        # rank_int[layer_id].append(int(r_float))
        flat_index.append(layer_id)
        # for gid, score in enumerate(fisher_list): # each group in one layer
        #     r_float = target_total_rank * score / fisher_sum # arrange rank base on group_score/all_score
        #     r_float = min(r_float, max_ranks_all_layers[layer_id][gid])  # 不超上限 make sure rank <= origianl rank
        #     rank_float[layer_id].append(r_float)
        #     rank_int[layer_id].append(int(r_float))
        #     flat_index.append((layer_id, gid))

    print("rank_float: ", rank_float)
    print("rank_int: ", rank_int)
    

    # --------------- Step 3. 贪心加余量 -----------------------------------------
    current = sum(sum(v) for v in rank_int.values())
    residual = target_total_rank - current

    print("residual: ", residual)

    # residual 按 (float - int) 大到小排序
    # flat_index.sort(key=lambda t: rank_float[t[0]][t[1]] - rank_int[t[0]][t[1]], reverse=True)
    flat_index.sort(key=lambda k: rank_float[k][0] - rank_int[k][0], reverse=True)
    print("flat_index: ", flat_index)
    # for layer_id, gid in flat_index:
    for layer_id in flat_index:
        max_r = sum(max_ranks_all_layers[layer_id])
        while rank_int[layer_id][0] < max_r and residual > 0:
            rank_int[layer_id][0] += 1
            residual -= 1
        if residual == 0:
            break
    # print("rank_int: ", dict(rank_int))
    # print(A)
    final_rank_sum = sum(sum(v) for v in rank_int.values())
    rank_int = split_values(dict(rank_int), 8)
    rank_int = rounding_search_result(rank_int)
    # print("rank_int: ", rank_int)
    # print(A)
    return rank_int, final_rank_sum, total_max_rank


from collections import defaultdict
from typing import Dict, List, Tuple
import math


def allocate_group_rank_by_size(
    layer_rank: dict,
    group_to_rows_all_layers: dict
) -> dict:
    """
    按照每个 group 的 head 数（大小）在该层的总 rank 中按比例分配。

    参数:
        layer_rank: Dict[str, int]，每层分配的 rank 数
        group_to_rows_all_layers: Dict[str, Dict[int, List[int]]]，每层的分组结构

    返回:
        Dict[str, List[int]]，每层的 group rank 分配
    """
    result = {}

    for layer_name, group_to_rows in group_to_rows_all_layers.items():
        total_heads = sum(len(heads) for heads in group_to_rows.values())
        total_rank = layer_rank[layer_name]

        # 初步按比例分配（浮点）
        group_sizes = [len(group_to_rows[g]) for g in sorted(group_to_rows)]
        raw_ranks = [total_rank * size / total_heads for size in group_sizes]
        floored_ranks = [math.floor(r) for r in raw_ranks]

        # 贪心修复误差
        residual = total_rank - sum(floored_ranks)
        remainders = [(i, raw_ranks[i] - floored_ranks[i]) for i in range(len(floored_ranks))]
        remainders.sort(key=lambda x: -x[1])  # 残差排序
        for i in range(residual):
            floored_ranks[remainders[i][0]] += 1

        result[layer_name] = floored_ranks

    return result

import math


def allocate_group_rank_evenly(
    layer_rank: dict,
    group_to_rows_all_layers: dict,
    max_ranks_all_layers: dict
) -> dict:
    """
    将每层的 rank 在 group 中均匀分配，限制每个 group 的分配不超过 max_ranks。

    参数:
        layer_rank: Dict[str, int]，每层分配的 rank 数
        group_to_rows_all_layers: Dict[str, Dict[int, List[int]]]，每层的分组结构
        max_ranks_all_layers: Dict[str, Dict[int, int]]，每层每个 group 最大允许的 rank

    返回:
        Dict[str, List[int]]，每层的 group rank 分配（按 group index 排序）
    """
    result = {}

    for layer_name, group_to_rows in group_to_rows_all_layers.items():
        group_keys = sorted(group_to_rows.keys())  # 保证顺序一致
        num_groups = len(group_keys)
        total_rank = layer_rank[layer_name]
        max_ranks = max_ranks_all_layers[layer_name]

        # 初始化：每个 group 先分 base_rank，多余部分后分
        base_rank = total_rank // num_groups
        remainder = total_rank % num_groups
        group_ranks = [base_rank] * num_groups
        for i in range(remainder):
            group_ranks[i] += 1

        # 第一轮修正：如果某个 group 分配超过 max_rank，就收回多余部分
        overflow = 0
        for i, gid in enumerate(group_keys):
            max_r = max_ranks[gid]
            if group_ranks[i] > max_r:
                overflow += group_ranks[i] - max_r
                group_ranks[i] = max_r

        # 第二轮重分：把收回的 rank 分配给还没到 max 的 group
        while overflow > 0:
            updated = False
            for i, gid in enumerate(group_keys):
                max_r = max_ranks[gid]
                if group_ranks[i] < max_r:
                    group_ranks[i] += 1
                    overflow -= 1
                    updated = True
                    if overflow == 0:
                        break
            if not updated:
                break  # 无法再分配

        result[layer_name] = group_ranks

    return result



def allocate_rank_from_group_rows_and_fisher_new(
    model,
    tokenizer,
    group_to_rows_all_layers: Dict[str, Dict[int, List[int]]],
    param_ratio_target: float
) -> Tuple[Dict[str, List[int]], int, int]:
    """
    return:
        select_result   : {layer_name: [rank_group0, rank_group1, ...]}
        final_rank_sum  : final rank sum
        total_max_rank  : original rank before compression
    """

    calib_loader = get_calib_data("wikitext2", tokenizer, "meta-llama/Llama-2-7b-hf", 2048, seqlen=1024)
    fisher_info_all_layers = calib_fisher_info(model, calib_loader, model.device, group_to_rows_all_layers, True)

    # ---------- Step 0: estimate the max rank of each group ----------
    max_ranks_all_layers = {}
    for layer_name, group_to_rows in group_to_rows_all_layers.items():
        num_heads_layer = sum(len(r) for r in group_to_rows.values())
        raw_out_features = detect_head_dim_from_name(model, layer_name)
        head_dim = raw_out_features // num_heads_layer
        max_ranks_all_layers[layer_name] = [len(r) * head_dim for r in group_to_rows.values()]

    

    # ---------- Step 1: compute total_max_rank ----------
    total_max_rank = sum(sum(v) for v in max_ranks_all_layers.values())
    target_total_rank = int(total_max_rank * param_ratio_target)
    print("total_max_rank: ", total_max_rank)
    print("target_total_rank: ", target_total_rank)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("max_ranks_all_layers: ", max_ranks_all_layers)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
    

    # ---------- Step 1.5: distribute layer rank number: target_rank ----------
    fisher_layer_sum = {name: sum(lst) for name, lst in fisher_info_all_layers.items()}
    fisher_total_sum = sum(fisher_layer_sum.values())
    print("fisher_layer_sum: ", fisher_layer_sum)

    layer_target_rank = {}
    layer_target_rank_float = {}

    for name, layer_fisher in fisher_layer_sum.items():
        rank_float = target_total_rank * layer_fisher / fisher_total_sum
        layer_target_rank[name] = min(sum(max_ranks_all_layers[name]), math.floor(rank_float))
        layer_target_rank_float[name] = rank_float

    # ---------- Step 1.5.1: greedy residue ----------
    indexes = []
    for name, i in fisher_layer_sum.items():
        indexes.append(name)

    indexes = sorted(indexes, key=lambda x: layer_target_rank_float[x] - layer_target_rank[x])
    dif = target_total_rank - sum(layer_target_rank.values())
    print(dif)
    print(indexes)

    while dif > 0:
        for name in indexes:  
            if layer_target_rank[name] >= 4096:
                continue
            layer_target_rank[name] += 1
            dif -= 1
            if dif == 0:
                break
    
    print("layer_target_rank after: ", layer_target_rank)
    # distribute by head size
    # result = allocate_group_rank_by_size(layer_target_rank, group_to_rows_all_layers)
    # Evenly
    result = allocate_group_rank_evenly(layer_target_rank, group_to_rows_all_layers, max_ranks_all_layers)


    final_rank_sum = sum(sum(v) for v in result.values())

    result = rounding_search_result(result)
    print("result: ", result)
    print("++++++++++++++++++++++++++")
    print(final_rank_sum/total_max_rank)
    # print(A)

    return result, final_rank_sum, total_max_rank






# test
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# from cluster_search import collect_headwise_KV, get_per_layer_group_to_rows, compute_similiarity
# K, V = collect_headwise_KV(model, tokenizer, model.device)
# print(len(K))
# K_dict = get_per_layer_group_to_rows(K, compute_similiarity, 0.7)
# # V_dict = get_per_layer_group_to_rows(V, compute_similiarity, 0.7)

# print(K_dict)
# new_dict = convert_group_to_rows_to_layername(model, K_dict)
# print(new_dict)

# select_result, final_rank, total_rank = allocate_rank_from_group_rows_and_fisher(model, tokenizer, K_dict, 0.7)
# select_result = convert_selection_result_to_layername(model, select_result)
# print("select_result: ", select_result)