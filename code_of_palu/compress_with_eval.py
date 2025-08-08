import argparse
import torch
import sys
from loguru import logger
from utils import set_seed, dump_to_huggingface_repos_new, load_model_and_tokenizer
from palu.rank_search import rank_search
from tqdm import tqdm
# from palu.decomposition import compress_model
from palu.cluster_search import collect_headwise_KV, get_per_layer_group_to_rows, compute_similiarity
from palu.rank_search_new import allocate_rank_from_group_rows_and_fisher_K, allocate_rank_from_group_rows_and_fisher_V, convert_group_to_rows_to_layername, convert_selection_result_to_layername, merge_kv_from_model_name, merge_kv_select_results
from palu.decomposition_new import compress_model
# from run_lm_eval import run_lm_eval_zero_shot
from run_ppl_eval import eval_ppl
import os

def compress(args):
    # set seed
    set_seed(args.seed)
    # load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    model.to(torch.device(args.device))
    K_list, V_list = collect_headwise_KV(model, tokenizer, model.device)
    K_dict = get_per_layer_group_to_rows(K_list, compute_similiarity, 0.6)
    V_dict = get_per_layer_group_to_rows(V_list, compute_similiarity, 0.6)
    del K_list, V_list
    print(K_dict)
    Dict = merge_kv_from_model_name(model, K_dict, V_dict)
    print(Dict)
    
    select_result_K, final_rank_K, total_rank_K = allocate_rank_from_group_rows_and_fisher_K(model, tokenizer, K_dict, 0.5)
    select_result_V, final_rank_V, total_rank_V = allocate_rank_from_group_rows_and_fisher_K(model, tokenizer, V_dict, 0.5)
    select_result = merge_kv_select_results(model, select_result_K, select_result_V)
    # select_result = convert_selection_result_to_layername(model, select_result)
    # K_dict = convert_group_to_rows_to_layername(model, K_dict)
    # print(K_dict)
    print("+++++++++++++")
    print(select_result)
    # Step 1: Perform rank selection to get layer-wise compression rate
    # search_results, rank_sum, total_rank = rank_search(model, tokenizer, args)
    # print(torch.cuda.memory_summary())
    # Step 2: Compress models
    compress_model(model, tokenizer, args, args.device, select_result, Dict)
    # compress_model(model, tokenizer, args, args.device, search_results)
    results = eval_ppl(model, tokenizer,model_name="llama2-7b", datasets="wikitext2", seqlen=2048, device="cuda")
    for dataset, ppl in results.items():
        logger.info(f"PPL: {ppl}")
    if args.dump_huggingface_model:
        save_folder = f"{args.model_id.split('/')[-1]}_ratio-{args.param_ratio_target}_gs-{args.head_group_size}-{args.search_method}-{args.decompose_method}"
        dump_to_huggingface_repos_new(model, tokenizer, save_folder, args)
        logger.info(f"Huggingface model is saved to {save_folder}", fg="green")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        # default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Pretrained model ID"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed"
    )

    parser.add_argument(
        "--dump_huggingface_model", 
        action="store_true",
        help="Whether to dump huggingface model or not."
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cached calibration results or not.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--n_fisher_calib_samples",
        type=int,
        default=32,
        help="Number of samples used for calibration.",
    )
    
    parser.add_argument(
        "--n_whiten_calib_samples",
        type=int,
        default=256,
        help="Number of samples used for calibration.",
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset",
    )

    parser.add_argument(
        "--calib_seqlen",
        type=int,
        default=1024,
        help="Sequence length of the calibration dataset."
    )

    parser.add_argument(
        "--head_group_size",
        type=int,
        default=4,
        help="Group size for group-wise decomposition."
    )


    # Rank Search hyper-paramters
    parser.add_argument(
        "--param_ratio_target", 
        type=float,
        default=-1,
        help="Target param ratio"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    
    parser.add_argument(
        "--search_method",
        type=str,
        default="fisher_uniform",
        choices=["fisher", "fisher_uniform", "uniform"],
        help="Search method",
    )
    
    parser.add_argument(
        '--decompose_method',
        type=str,
        default='svd_cal_sim',
        choices=['whiten', 'svd', 'svd_cal', 'svd_cal_sim'],
        help='Decomposition method'
    )
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO" if not args.verbose else "DEBUG")
    
    compress(args)