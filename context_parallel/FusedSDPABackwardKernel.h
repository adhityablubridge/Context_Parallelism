#pragma once

// ---------------------------------------------------------------------------
// FusedSDPABackwardKernel.h
//
// Declares the launcher for the fused FlashAttention backward pass.
//
// Computes dQ, dK, dV without materialising the full [T_q x T_k] attention
// matrix in HBM.  Two kernels are launched sequentially in stream 0:
//
//   Kernel 1 (dQ)   : grid (ceil(T_q/32), BH), block 32
//                     Sweeps K/V tiles; writes dQ and D_buf[BH,T_q].
//
//   Kernel 2 (dK/V) : grid (ceil(T_k/32), BH), block 32
//                     Sweeps Q tiles; reads D_buf; writes dK, dV.
//
// D_buf [BH * T_q] must be caller-allocated (float, device memory).
//
// LSE must be the MERGED log-sum-exp [BH * T_q] (flat, no keepdim dim).
// For context-parallel ring attention steps, this is the merged LSE across
// all ring steps -- P_ij = exp(s_ij - merged_lse_i) directly.
//
// Supported HEAD_DIM: 32, 64, 128.
// ---------------------------------------------------------------------------

void launch_flash_attn_bwd_f32(
    const float *Q,   // [BH, T_q, D]
    const float *K,   // [BH, T_k, D]
    const float *V,   // [BH, T_k, D]
    const float *O,   // [BH, T_q, D]  -- merged forward output
    const float *dO,  // [BH, T_q, D]  -- incoming gradient
    const float *LSE, // [BH, T_q]     -- merged log-sum-exp (flat)
    float *dQ,        // [BH, T_q, D]  -- output gradient for Q
    float *dK,        // [BH, T_k, D]  -- output gradient for K
    float *dV,        // [BH, T_k, D]  -- output gradient for V
    float *D_buf,     // [BH, T_q]     -- scratch (caller-allocated)
    int BH, int T_q, int T_k, int D, float scale, bool is_causal, int q_offset,
    int k_offset);
