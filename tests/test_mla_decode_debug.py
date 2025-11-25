#!/usr/bin/env python3
"""
Test script for debugging mla_decode_fwd_grouped with saved tensors.

This script loads the saved tensors when kv_indptr=[0, 8] and runs 
the mla_decode_fwd_grouped operator to reproduce the error.

Usage:
    python tests/test_mla_decode_debug.py
"""

import os
import torch
import pytest


def test_mla_decode_fwd_grouped_with_saved_tensors():
    """Test mla_decode_fwd_grouped with saved tensors from the error case."""
    
    # Load saved tensors
    save_path = "/tmp/mla_decode_debug/mla_decode_inputs.pt"
    
    if not os.path.exists(save_path):
        pytest.skip(f"Saved tensors not found at {save_path}. Run vLLM to generate them first.")
    
    print(f"Loading tensors from {save_path}")
    data = torch.load(save_path)
    
    # Move tensors to GPU
    device = torch.device("cuda:0")
    q = data['q'].to(device)
    k_buffer = data['k_buffer'].to(device)

    o = data['o'].to(device)
    kv_indptr = data['kv_indptr'].to(device)
    block_tables = data['block_tables'].to(device)
    attn_logits = data['attn_logits'].to(device)
    attn_lse = data['attn_lse'].to(device)
    
    # Scalar parameters
    kv_lora_rank = data['kv_lora_rank']
    num_kv_splits = data['num_kv_splits']
    sm_scale = data['sm_scale']
    logit_cap = data['logit_cap']
    mtp = data['mtp']
    num_heads = data['num_heads']
    batch_size = data['batch_size']
    
    # Print debug info
    print("\n" + "=" * 80)
    print("Loaded tensor shapes:")
    print("=" * 80)
    print(f"q.shape: {q.shape}, dtype: {q.dtype}")
    print(f"k_buffer.shape: {k_buffer.shape}, dtype: {k_buffer.dtype}")
    print(f"v_buffer.shape: {v_buffer.shape}, dtype: {v_buffer.dtype}")
    print(f"o.shape: {o.shape}, dtype: {o.dtype}")
    print(f"kv_indptr: {kv_indptr}")
    print(f"kv_indices.shape: {kv_indices.shape}")
    print(f"kv_indices: {kv_indices}")
    print(f"block_tables.shape: {block_tables.shape}")
    print(f"block_tables[0, :10]: {block_tables[0, :10]}")
    print(f"attn_logits.shape: {attn_logits.shape}")
    print(f"attn_lse.shape: {attn_lse.shape}")
    print(f"\nParameters:")
    print(f"  kv_lora_rank: {kv_lora_rank}")
    print(f"  num_kv_splits: {num_kv_splits}")
    print(f"  sm_scale: {sm_scale}")
    print(f"  logit_cap: {logit_cap}")
    print(f"  mtp: {mtp}")
    print(f"  num_heads: {num_heads}")
    print(f"  batch_size: {batch_size}")
    print("=" * 80)
    
    # Import the operator
    try:
        from vllm import _rocm_aiter_ops as rocm_aiter_ops
    except ImportError:
        pytest.skip("rocm_aiter_ops not available")
    
    # Run the operator
    print("\nCalling mla_decode_fwd_grouped...")
    try:
        rocm_aiter_ops.mla_decode_fwd_grouped(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            o=o,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            block_tables=block_tables,
            kv_lora_rank=kv_lora_rank,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            mtp=mtp,
        )
        torch.cuda.synchronize()
        print("✓ mla_decode_fwd_grouped completed successfully!")
        print(f"Output o.shape: {o.shape}")
        print(f"Output o (first 10 elements): {o.flatten()[:10]}")
        
    except Exception as e:
        print(f"✗ mla_decode_fwd_grouped failed with error:")
        print(f"  {type(e).__name__}: {e}")
        raise


def test_analyze_kv_indices():
    """Analyze the kv_indices to understand the pattern."""
    
    save_path = "/tmp/mla_decode_debug/mla_decode_inputs.pt"
    
    if not os.path.exists(save_path):
        pytest.skip(f"Saved tensors not found at {save_path}")
    
    print(f"Loading tensors from {save_path}")
    data = torch.load(save_path)
    
    kv_indptr = data['kv_indptr']
    kv_indices = data['kv_indices']
    block_tables = data['block_tables']
    
    print("\n" + "=" * 80)
    print("Analyzing kv_indices pattern:")
    print("=" * 80)
    print(f"kv_indptr: {kv_indptr}")
    print(f"kv_indices.shape: {kv_indices.shape}")
    print(f"kv_indices: {kv_indices}")
    print(f"\nblock_tables.shape: {block_tables.shape}")
    print(f"block_tables[0]: {block_tables[0]}")
    
    # Analyze
    num_indices = kv_indptr[1] - kv_indptr[0]
    print(f"\nNumber of indices for request 0: {num_indices}")
    print(f"Unique values in kv_indices: {torch.unique(kv_indices)}")
    print(f"Value counts:")
    for val in torch.unique(kv_indices):
        count = (kv_indices == val).sum()
        print(f"  {val}: {count} occurrences")
    
    # Check for issues
    if (kv_indices == 0).sum() > 1:
        print("\n⚠ WARNING: Multiple indices pointing to block 0!")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run without pytest
    print("Testing mla_decode_fwd_grouped with saved tensors...")
    
    try:
        test_analyze_kv_indices()
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    try:
        test_mla_decode_fwd_grouped_with_saved_tensors()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

