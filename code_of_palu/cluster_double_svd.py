import os, click
import torch
import torch.nn as nn
from loguru import logger
# from .model import AVAILABLE_MODELS
# from .data_utils import get_calib_data
import math
from tqdm import tqdm
import json


# def collect_headwise_KV(model, tokenizer, device):
    
#     # calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)
    
#     # We'll collect outputs for all layers in these lists
#     collected_k_outputs = []
#     collected_v_outputs = []
    
#     def k_proj_hook(module, input, output):
#         """
#         module: The layer that produced this output (k_proj).
#         input:  The input to k_proj.
#         output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
#         """
#         B, S, D = output.shape
#         H = 8
#         head_dim = D // H
#         output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
#         # Detach to avoid growing the autograd graph
#         collected_k_outputs.append(output.detach().cpu())

#     def v_proj_hook(module, input, output):
#         """
#         module: The layer that produced this output (v_proj).
#         input:  The input to v_proj.
#         output: The output from v_proj (shape [batch_size, seq_len, hidden_dim]).
#         """
#         B, S, D = output.shape
#         H = 32
#         head_dim = D // H
#         output = output.view(B, S, H, head_dim).transpose(1, 2).squeeze(0) # [num_head, seq_len, head_dim]
#         # Detach to avoid growing the autograd graph
#         collected_v_outputs.append(output.detach().cpu())

#     num_layers = len(model.model.layers)
#     hooks_k = []
#     hooks_v = []
#     for layer_idx in range(num_layers):
#         # Access the i-th layer
#         layer = model.model.layers[layer_idx].self_attn
#         # print(f"  - K/V heads: {layer.config.num_key_value_heads}")
        
#         # Register forward hooks
#         hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
#         hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
        
#         hooks_k.append(hook_k)
#         hooks_v.append(hook_v)


#     model.eval()
#     model.to(device)
#     with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama3.json", "r", encoding="utf-8") as f:
#         prompts = json.load(f)
#     prompt_2k = prompts["2k"]
#     inputs = tokenizer(prompt_2k, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     for hook in hooks_k:
#         hook.remove()
#     for hook in hooks_v:
#         hook.remove()
    
#     return collected_k_outputs, collected_v_outputs


# # test
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# K, V = collect_headwise_KV(model, tokenizer, model.device)
# print(K[0].shape)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from types import MethodType
import json
import matplotlib.pyplot as plt

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
def collect_KV(model, tokenizer):

    with open("/home/azzhang/streaming-llm/output/wikitext2_prompts_llama2.json", "r", encoding="utf-8") as f:
        prompts = json.load(f)

    target_length_4k = "4k"


    prompt_4k = prompts[target_length_4k]
    inputs_4k = tokenizer(prompt_4k, return_tensors="pt").to(model.device)
    # seq_len_4 = inputs_4k["input_ids"].shape[1]

    cache_4k = {}

    # ✅ 提前 patch 所有层
    for layer_idx in range(32):
        def make_patched_forward(layer_id):
            # def patched_forward(self, hidden_states, attention_mask=None, position_embeddings=None, position_ids=None, past_key_value=None, output_attentions=False, **kwargs):
            def patched_forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_value=None, output_attentions=False, use_cache=False,
                        **kwargs):
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)

                bsz, seqlen, dim = k.shape
                head_dim = self.head_dim
                num_heads_q = self.config.num_attention_heads
                num_heads_kv = self.config.num_key_value_heads

                q = q.view(bsz, seqlen, num_heads_q, head_dim).transpose(1, 2)
                k = k.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)
                v = v.view(bsz, seqlen, num_heads_kv, head_dim).transpose(1, 2)

                # cos, sin = position_embeddings
                cos, sin = self.rotary_emb(v, position_ids.shape[1])  # 注意这里是 value_states 用于生成位置编码
                q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                layer_name = f"model.layers.{layer_id}.self_attn"
                cache_4k[f"{layer_name}.k_proj"] = k_rope.detach().cpu()
                cache_4k[f"{layer_name}.v_proj"] = v.detach().cpu()

                # return self._orig_forward(hidden_states, attention_mask, position_embeddings, position_ids, past_key_value, output_attentions, **kwargs)
                return self._orig_forward(hidden_states, attention_mask, position_ids,
                                past_key_value, output_attentions, use_cache,
                                **kwargs)
            return patched_forward

        attn_layer = model.model.layers[layer_idx].self_attn
        attn_layer._orig_forward = attn_layer.forward
        attn_layer.forward = MethodType(make_patched_forward(layer_idx), attn_layer)

    # ✅ 一次 forward 就能抓到所有层的 K/V
    with torch.no_grad():
        _ = model(**inputs_4k)

    return cache_4k

# cache = collect_KV(model, tokenizer)
# print(cache)