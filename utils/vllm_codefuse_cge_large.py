# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Tuple
from vllm.sequence import EmbeddingSequenceGroupOutput, PoolerOutput
import torch
from torch import nn
from transformers import Qwen2Config
from transformers import PretrainedConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput

from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import is_pp_missing_parameter, make_layers
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import PoolerOutput
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen2DecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config),
            prefix=f"{prefix}.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(config, cache_config, quant_config)

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

class CodeFuse_CGE_Large(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = %s is less than "
                             "`num_hidden_layers` = %s. Please open an issue "
                             "to discuss this feature." % (
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

        super().__init__()

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.plm_model = Qwen2ForCausalLM(config, cache_config, quant_config)
        self.embedding_method = config.embedding_method
        self.inf_seq_length = config.inf_seq_length
        self.padding_side = config.padding_side
        self.keep_max_layer = config.keep_max_layer
        self.emb_dim = self.plm_model.model.embed_tokens.weight.size(1)
        self.num_heads = config.pma_num_heads
        self.ln = config.pma_ln
        self.norm = config.pma_norm
        self.pma_mode = config.pma_norm_mode
        self.mha_pma = PMA(self.emb_dim, self.compress_dim, self.num_heads, 1, ln=self.ln, pma_mode=self.pma_mode).to("cuda")
        if config.tie_word_embeddings:
            self.lm_head = self.plm_model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        for param_tensor in self.mha_pma.state_dict():
            print(param_tensor, "\t", self.mha_pma.state_dict()[param_tensor])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.plm_model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        
        embedding = hidden_states.unsqueeze(0)
        res_embedding = self.pma_embedding(embedding, positions.unsqueeze(0))
        return res_embedding

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        hidden_states = nn.functional.normalize(hidden_states, p=2, dim=1)
        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in hidden_states
        ]

        return PoolerOutput(outputs=pooled_outputs)

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        for param_tensor in self.mha_pma.state_dict():
            print(param_tensor, "\t", self.mha_pma.state_dict()[param_tensor])

    def last_embedding(self, A, index):
        bs, seq, emb = A.size()
        res = A[torch.arange(bs), index, :]
        return res

    def mean_embedding(self, A, mask):
        bs, seq, emb = A.size()
        res = (A * (mask.unsqueeze(-1))).sum(1) / (mask.sum(1).unsqueeze(-1))
        return res
    
    # A (bs, seq, emb_size), mask (bs, 1, seq)
    def weighted_embedding(self, A, mask):
        weights = (torch.arange(start=1, end=A.size(1) + 1).unsqueeze(0).unsqueeze(-1).expand(A.size()).float()).to(A.device)
        input_mask_expanded = (mask.squeeze(1).unsqueeze(-1).expand(A.size()).float()).to(A.device)
        sum_embedding = torch.sum(A * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        weighted_embedding = sum_embedding / sum_mask
        
        return weighted_embedding

    def pma_embedding(self, A, mask):
        res = self.mha_pma(A, mask).squeeze(1)
        return res


    def get_sentence_embedding(self, embedding_method, **inputs):
        outputs = self.plm_model(inputs['input_ids'], inputs['attention_mask'], output_hidden_states=True)
        if embedding_method == 'last':
            embedding = outputs.hidden_states[self.keep_max_layer]
            index = inputs['attention_mask'].sum(-1).long() - 1
            res_embedding = self.last_embedding(embedding, index)
        elif embedding_method == 'mean':
            embedding = outputs.hidden_states[self.keep_max_layer]
            res_embedding = self.mean_embedding(embedding, inputs['attention_mask'])
        elif embedding_method == 'weighted':
            embedding = outputs.hidden_states[self.keep_max_layer]
            res_embedding = self.weighted_embedding(embedding, inputs['attention_mask'])
        elif embedding_method == 'pma':
            embedding = outputs.hidden_states[self.keep_max_layer]
            attention_mask = inputs['attention_mask']
            res_embedding = self.pma_embedding(embedding, attention_mask)
        else:
            logger.debug('Error, no {} way to obtain embbedings'.format(embedding_method))
        
        if not self.norm:
            res_embedding = torch.nn.functional.normalize(res_embedding, p=2.0, dim=-1, eps=1e-12, out=None)
        return res_embedding



    def encode(self, tokenizer, sentences, batch_size=32, convert_to_numpy=True,
            convert_to_tensor=False, show_progress_bar=True, max_seq_length=None, **kwargs):
        if max_seq_length is None:
            max_seq_length = self.inf_seq_length
        input_is_string = False        
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx] # 大到小重排
        with torch.no_grad():
            for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
                sentences_batch = sentences_sorted[start_index: start_index + batch_size]
                # Compute sentences embeddings
                with torch.no_grad():
                    inputs = tokenizer(sentences_batch, padding=True, truncation=True, max_length=max_seq_length, add_special_tokens=False, return_tensors='pt').to(self.plm_model.device)
                    embeddings = self.get_sentence_embedding(self.embedding_method, **inputs)
                embeddings = embeddings.detach()
                if convert_to_numpy:
                    if embeddings.dtype == torch.bfloat16:
                        embeddings = embeddings.cpu().to(torch.float32)
                    else:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings


class MAB_POST(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB_POST, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

    def forward(self, Q, K, pad_mask=None):

        Q_ = self.fc_q(Q)
        K_, V_ = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)

        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
        score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = Q + O
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class MAB_PRE_NORMAL(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB_PRE_NORMAL, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln_q = nn.LayerNorm(dim_V)
            self.ln_kv = nn.LayerNorm(dim_V)
            self.ln_o = nn.LayerNorm(dim_V)
            self.ln_final = nn.LayerNorm(dim_V)
        
        self.fc_o = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

    def forward(self, Q, K, pad_mask=None):
        Q_ = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K_ = K if getattr(self, 'ln_kv', None) is None else self.ln_kv(K)
        Q_ = self.fc_q(Q_) 
        K_, V_ = self.fc_k(K_), self.fc_v(K_)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)
        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
        score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2) 
        O = Q + O
        O_ = O if getattr(self, 'ln_o', None) is None else self.ln_o(O)
        O_ = O + F.relu(self.fc_o(O_))
        return O_ if getattr(self, 'ln_final', None) is None else self.ln_final(O_)


class MAB_PRE_GPTJ(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB_PRE_GPTJ, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        if ln:
            self.ln_q = nn.LayerNorm(dim_V)
            self.ln_kv = nn.LayerNorm(dim_V)
            self.ln_final = nn.LayerNorm(dim_V)

    def forward(self, Q, K, pad_mask=None):
        Q_ = Q if getattr(self, 'ln_q', None) is None else self.ln_q(Q)
        K_ = K if getattr(self, 'ln_kv', None) is None else self.ln_kv(K)

        Q1 = self.fc_q(Q_) 
        K1, V1 = self.fc_k(K_), self.fc_v(K_)
        dim_split = self.dim_V // self.num_heads

        Q1 = torch.cat(Q1.split(dim_split, 2), 0)
        K1 = torch.cat(K1.split(dim_split, 2), 0)
        V1 = torch.cat(V1.split(dim_split, 2), 0)


        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
        score = Q1.bmm(K1.transpose(1,2))/math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O1 = torch.cat(A.bmm(V1).split(Q.size(0), 0), 2)
        O2 = F.relu(self.fc_o(Q_))
        O_final = Q + O1 + O2
        return O_final if getattr(self, 'ln_final', None) is None else self.ln_final(O_final)


class PMA(nn.Module):
    def __init__(self, dim, compress_dim, num_heads, num_seeds, ln=False, pma_mode=None):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, compress_dim))
        nn.init.xavier_uniform_(self.S)
        if pma_mode == 'post_normal':
            self.mab = MAB_POST(compress_dim, dim, compress_dim, num_heads, ln=ln)
        elif pma_mode == 'pre_normal':
            self.mab = MAB_PRE_NORMAL(compress_dim, dim, compress_dim, num_heads, ln=ln)
        elif pma_mode == 'pre_gptj':
            self.mab = MAB_PRE_GPTJ(compress_dim, dim, compress_dim, num_heads, ln=ln)
        else:
            raise ValueError(f"Error, the pma_mode {pma_mode} is not implemented !")

    def forward(self, X, pad_mask):
        if self.S.dtype != torch.bfloat16:
            X = X.float()
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)
