
import os, click
import torch
import torch.nn as nn
from loguru import logger
from .model import AVAILABLE_MODELS
from .data_utils import get_calib_data
import math
from tqdm import tqdm

def rounding_search_result(config: dict, block_size=32):
    for module_name in config.keys():
        ranks = config[module_name]
        for i in range(len(ranks)):
            ranks[i] = max(1, round(ranks[i] / block_size)) * block_size
        config[module_name] = ranks
    return config

def replace_with_mean(data):
    result = {}
    for key, value in data.items():
        if value:
            mean_value = sum(value) / len(value)
            new_value = [mean_value] * len(value)
            result[key] = new_value
    return result

def split_values(data, group_number):
    result = {}
    for key, value in data.items():
        new_value = [v // group_number for v in value for _ in range(group_number)]
        result[key] = new_value
    return result

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
# def allocate_group_rank_evenly(
#     layer_rank: dict,
#     group_to_rows_all_layers: dict
# ) -> dict:
#     """
#     将每层的 rank 均匀分配给每个 group（不考虑 head 数）。

#     参数:
#         layer_rank: Dict[str, int]，每层分配的 rank 数
#         group_to_rows_all_layers: Dict[str, Dict[int, List[int]]]，每层的分组结构

#     返回:
#         Dict[str, List[int]]，每层的 group rank 分配（按 group index 排序）
#     """
#     result = {}

#     for layer_name, group_to_rows in group_to_rows_all_layers.items():
#         group_keys = sorted(group_to_rows.keys())  # 保证顺序一致
#         num_groups = len(group_keys)
#         total_rank = layer_rank[layer_name]

#         base_rank = total_rank[0] // num_groups
#         remainder = total_rank[0] % num_groups

#         # 初始化每个 group 的 rank 为 base_rank
#         group_ranks = [base_rank] * num_groups

#         # 将剩余的 rank 均匀分给前几个 group
#         for i in range(remainder):
#             group_ranks[i] += 1

#         result[layer_name] = group_ranks

#     return result

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
        total_rank = layer_rank[layer_name][0]
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


def calib_fisher_info(model, calib_loader, device, use_cache=True):
    model.half()
    model.to(device)
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"

    logger.info(f"[Fisher] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and use_cache:
        logger.info(f"[Fisher] File {cache_file} exist.", fg="green")
        logger.info(f"[Fisher] Load cache_file={cache_file}", fg="yellow")
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    logger.info(f"[Fisher] No cache_file={cache_file}", fg="red")
    logger.info(f"[Fisher] Create fisher info list...", fg="yellow")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        # print(model.device)
        # print(input_ids.device)
        # print(labels.device)
        for name, param in model.named_parameters():
            if param.device != torch.device("cuda:0"):
                print(f"[Warning] Param {name} is on {param.device}")

        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                module.fisher_info += module.weight.grad.detach().to(torch.float32).pow(2)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "attn" in name:
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info

    logger.info(f"[Fisher] Save the fisher info list to:  {cache_file}", fg="yellow")
    torch.save(all_fisher_info, cache_file)

def rank_search(model: nn.Module, tokenizer, group_to_rows_all_layers, args):
    logger.info(f"[Rank search] Do rank searching. Search method: {args.search_method}", fg="yellow")
    if args.search_method == "uniform":
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        select_result = {}
        info = target_model_class.get_kv_info(model, args.head_group_size)
        
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:                
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims*args.param_ratio_target] * info.num_lr_groups})

        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        return select_result, rank_sum, total_rank    
    elif args.search_method == "fisher":
        # Prepare Fisher information
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
        total_rank = 0
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        
        info = target_model_class.get_kv_info(model, args.head_group_size)
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                if not torch.isfinite(fisher).all():
                    logger.info(fisher)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)


        target_rank = total_rank * args.param_ratio_target
        
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                rank_float = target_rank * fisher[i] / fisher_sum
                
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

            select_result_float.update({name: ranks})
                
        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        dif = target_rank - sum([sum(v) for k, v in select_result.items()])


        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break
                
        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        
        return select_result, rank_sum, total_rank    
    elif args.search_method == "fisher_uniform":
        # Prepare Fisher information
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
        calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
            
        total_rank = 0
        
        fisher_sum = 0.0
        fisher_info_dict = {}
        select_result = {}
        info = target_model_class.get_kv_info(model, model.config.num_key_value_heads)
        # print("info: ", info)

        max_ranks_all_layers = {}
        for layer_name, group_to_rows in group_to_rows_all_layers.items():
            num_heads_layer = sum(len(r) for r in group_to_rows.values())
            raw_out_features = detect_head_dim_from_name(model, layer_name)
            head_dim = raw_out_features // num_heads_layer
            max_ranks_all_layers[layer_name] = [len(r) * head_dim for r in group_to_rows.values()]
        
        for name, module in model.named_modules():
            if "k_proj" in name or "v_proj" in name:
                module_rank = info.num_lr_groups * info.lr_group_dims
                total_rank += module_rank
                
                select_result.update({name: [info.lr_group_dims] * info.num_lr_groups})
                fisher = module.fisher_info.reshape(info.num_lr_groups, -1, module.in_features)
                
                fisher_list = [torch.mean(fisher[i]).item() for i in range(info.num_lr_groups)]
                fisher_info_dict.update({name: fisher_list})
                fisher_sum += sum(fisher_list)

        # print("Total rank: ", total_rank)
        # print("fisher_info_dict: ", fisher_info_dict)
        # print("+++++++++++++++++++++++++++++")
        # print("fisher_sum: ", fisher_sum)

        target_rank = total_rank * args.param_ratio_target
        print("target_rank: ", target_rank)
        indexes = []
        select_result_float = {}

        for name, fisher in fisher_info_dict.items():
            ranks = []
            for i in range(len(fisher)):
                rank_float = target_rank * fisher[i] / fisher_sum    
                ranks.append(rank_float)
                indexes.append((name, i))
                select_result[name][i] = min(select_result[name][i], math.floor(rank_float))

            select_result_float.update({name: ranks})
        
        # print("Before greedy: ", select_result)
        # print("+++++++++++++++++++++++++++++")
        # print("Before greedy: ", select_result_float)
        # print("+++++++++++++++++++++++++++++")
        # print("indexes: ", indexes)

        indexes = sorted(indexes, key=lambda x: select_result_float[x[0]][x[1]] - select_result[x[0]][x[1]])
        dif = target_rank - sum([sum(v) for k, v in select_result.items()])
        print(select_result[indexes[0][0]][indexes[0][1]])
        print("+++++++++++++++++++++++++++++")
        print("indexes: ", indexes)

        while dif > 0:
            for i in range(len(indexes)):
                if select_result[indexes[i][0]][indexes[i][1]] == info.lr_group_dims:
                    continue
                select_result[indexes[i][0]][indexes[i][1]] += 1
                dif -= 1

                if dif == 0:
                    break

        print("Before split: ", select_result)
        print("+++++++++++++++++++++++++++++")
        
        # select_result = split_values(select_result, model.config.num_key_value_heads//args.head_group_size)
        select_result = allocate_group_rank_evenly(select_result, group_to_rows_all_layers, max_ranks_all_layers)
        print("Before rounding: ", select_result)
        print("+++++++++++++++++++++++++++++")
        select_result = rounding_search_result(select_result)
        rank_sum = sum([sum(v) for k, v in select_result.items()])
        logger.info(f"[Rank Search] KV-Cache Compression Ratio: {100-(rank_sum / total_rank * 100): .2f}%")
        print("After rounding: ", select_result)
        print("+++++++++++++++++++++++++++++")
        
        return select_result, rank_sum, total_rank
    else:
        raise NotImplementedError  