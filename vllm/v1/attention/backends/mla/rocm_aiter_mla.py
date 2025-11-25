# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.backends.abstract import AttentionLayer, MultipleOf
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.v1.attention.backends.utils import AttentionCGSupport
from vllm.v1.kv_cache_interface import AttentionSpec


class AiterMLABackend(MLACommonBackend):
    supported_kernel_block_sizes: ClassVar[list[int | MultipleOf]] = [1, 64]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None


class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    pass


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec, layer_names, vllm_config, device, AiterMLAMetadata
        )

        self.compilation_config = vllm_config.compilation_config
        # kernel block size is always 1.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )
            self.paged_kv_indices = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self.paged_kv_last_page_len = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.arange(
                0, max_num_reqs + 1, dtype=torch.int32, device=device
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        # kernel block size is always 1, although the kv block size is not 1.
        device = self.device
        num_reqs = seq_lens_device.size(0)
 
        mask = torch.arange(
            block_table_tensor.size(1), dtype=block_table_tensor.dtype, device=device
        ).unsqueeze(0) < seq_lens_device.unsqueeze(1)
            
        paged_kv_indices = block_table_tensor[mask]
        paged_kv_last_page_len = torch.where(seq_lens_device == 0, 1, seq_lens_device)

        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=seq_lens_device.dtype, device=device),
                seq_lens_device.cumsum(dim=0, dtype=torch.int32),
            ]
        )

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            num_actual_pages = paged_kv_indices.size(0)

            self.paged_kv_indices[:num_actual_pages].copy_(
                paged_kv_indices, non_blocking=True
            )
            self.paged_kv_indices[num_actual_pages:].fill_(-1)
            paged_kv_indices = self.paged_kv_indices[:num_actual_pages]

            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            self.paged_kv_last_page_len[:num_reqs].copy_(
                paged_kv_last_page_len, non_blocking=True
            )
            self.paged_kv_last_page_len[num_reqs:].fill_(1)
            paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            qo_indptr = torch.arange(
                0, num_reqs + 1, step=1, dtype=torch.int32, device=device
            )

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

        return attn_metadata


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        assert num_heads in [8, 16, 128], (
            f"Aiter MLA only supports 8, 16 or 128 number of heads.\n"
            f"Provided {num_heads} number of heads.\n"
            "Try adjusting tensor_parallel_size value."
        )
        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )
        return output

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        o = torch.zeros(
            B, self.num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )

        # max_seqlen_qo must be 1 except for MTP
        # TODO: Find the best value for MTP
        max_seqlen_qo = 1
        
        # For num_heads == 8, use decode_attention_fwd_grouped interface
        if self.num_heads == 8:
            # MLA KV cache shape: [num_blocks, block_size, head_size]
            # where head_size = kv_lora_rank + qk_rope_head_dim (e.g., 576 = 512 + 64)
            # Target: [num_blocks, block_size, 1, head_size] to add num_kv_heads dim
            num_blocks = kv_c_and_k_pe_cache.shape[0]
            block_size = kv_c_and_k_pe_cache.shape[1]
            head_size = kv_c_and_k_pe_cache.shape[2]
            
            # Add num_kv_heads dimension (1 for MLA)
            kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)  # [num_blocks, block_size, 1, head_size]

            # Prepare intermediate tensors
            # Set num_kv_splits to 1 for simple decode scenarios
            num_kv_splits = 8
            attn_logits = torch.empty(
                B,
                num_kv_splits,
                self.num_heads,
                self.kv_lora_rank,
                dtype=torch.float32,
                device=q.device,
            )
            attn_lse = torch.empty(
                B,
                num_kv_splits,
                self.num_heads,
                1,
                dtype=torch.float32,
                device=q.device,
            )

            # Reshape q to [batch, num_heads, head_size]
            q_reshaped = q.view(B, self.num_heads, -1)
            from aiter import dtypes
       
            # 只在 rank 0 打印调试信息
            if parallel_state.get_world_group().is_first_rank:
                print("=" * 80, flush=True)
                print("mla_decode_fwd_grouped 参数信息:", flush=True)
                print("=" * 80, flush=True)
                print(f"After reshape - kv_buffer.shape: {kv_buffer.shape}", flush=True)
                print(f"  num_blocks: {num_blocks}, block_size: {block_size}, head_size: {head_size}", flush=True)
                print(f"q_reshaped.shape: {q_reshaped.shape}, dtype: {q_reshaped.dtype}", flush=True)
                print(f"kv_buffer.shape: {kv_buffer.shape}, dtype: {kv_buffer.dtype}", flush=True)
    
                print(f"o.shape: {o.shape}, dtype: {o.dtype}", flush=True)
                
                # kv_indptr - 打印 shape, dtype 和值
                kv_indptr = attn_metadata.decode.paged_kv_indptr
                print(f"\nkv_indptr.shape: {kv_indptr.shape}, dtype: {kv_indptr.dtype}", flush=True)
                if kv_indptr.numel() <= 100:
                    print(f"kv_indptr values: {kv_indptr}", flush=True)
                else:
                    print(f"kv_indptr values (first 10): {kv_indptr[:10]}", flush=True)
                    print(f"kv_indptr values (last 10): {kv_indptr[-10:]}", flush=True)
                
                print(f"\nblock_tables.shape: {attn_metadata.decode.block_table.shape}, dtype: {attn_metadata.decode.block_table.dtype}", flush=True)
                print(f"block_tables[0, :10]: {attn_metadata.decode.block_table[0, :10]}", flush=True)
                print(f"attn_logits.shape: {attn_logits.shape}, dtype: {attn_logits.dtype}", flush=True)
                print(f"attn_lse.shape: {attn_lse.shape}, dtype: {attn_lse.dtype}", flush=True)
                print(f"\nkv_lora_rank: {self.kv_lora_rank}", flush=True)
                print(f"num_kv_splits: {num_kv_splits}", flush=True)
                print(f"sm_scale: {self.scale}", flush=True)
                print(f"logit_cap: 0.0", flush=True)
                print(f"mtp: {max_seqlen_qo - 1}", flush=True)
                print(f"\nBatch size (B): {B}", flush=True)
                print(f"num_heads: {self.num_heads}", flush=True)
                print("=" * 80, flush=True)
            rocm_aiter_ops.mla_decode_fwd_grouped(
                q=q_reshaped,
                k_buffer=kv_buffer,
                v_buffer=kv_buffer,
                o=o,
                kv_indptr=attn_metadata.decode.paged_kv_indptr,
                block_tables=attn_metadata.decode.block_table,
                kv_lora_rank=self.kv_lora_rank,
                attn_logits=attn_logits,
                attn_lse=attn_lse,
                num_kv_splits=num_kv_splits,
                sm_scale=self.scale,
                logit_cap=0.0,
                mtp=max_seqlen_qo - 1,
            )
            torch.cuda.synchronize()
        else:
            # For num_heads == 16 or 128, use the original mla_decode_fwd
            kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

            rocm_aiter_ops.mla_decode_fwd(
                q,
                kv_buffer,
                o,
                self.scale,
                attn_metadata.decode.qo_indptr,
                max_seqlen_qo,
                attn_metadata.decode.paged_kv_indptr,
                attn_metadata.decode.paged_kv_indices,
                attn_metadata.decode.paged_kv_last_page_len,
            )

        return o, None
