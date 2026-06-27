#pragma once

// ---------------------------------------------------------------------------
// FusedSDPAKernel.h
//
// Declares the CUDA kernel launcher for fused FlashAttention-style forward.
//
// The kernel implements the online-softmax tile algorithm from FlashAttention
// (Dao et al. 2022): attention scores, softmax, and V-weighted sum are all
// computed inside shared/register memory without writing the full [T_q x T_k]
// attention matrix to HBM.
//
// Inputs  (float*, contiguous, layout [BH, T, D] where BH = B*H):
//   Q    [BH, T_q, D]
//   K    [BH, T_k, D]
//   V    [BH, T_k, D]
//
// Outputs:
//   O    [BH, T_q, D]   attention output
//   LSE  [BH, T_q]      log-sum-exp per query row (for SDPAMerger)
//
// Parameters:
//   BH         - batch * heads (merged outer dim)
//   T_q        - number of query tokens in this chunk
//   T_k        - number of key/value tokens in this chunk
//   D          - per-head dimension (must be <= 128)
//   scale      - attention scale (typically 1/sqrt(D))
//   is_causal  - apply upper-triangular causal mask
//   q_offset   - global sequence start position of Q chunk (for causal mask)
//   k_offset   - global sequence start position of K chunk (for causal mask)
// ---------------------------------------------------------------------------

void launch_flash_attn_fwd_f32(
    const float* Q,
    const float* K,
    const float* V,
    float*       O,
    float*       LSE,
    int BH, int T_q, int T_k, int D,
    float scale,
    bool  is_causal,
    int   q_offset,
    int   k_offset);
