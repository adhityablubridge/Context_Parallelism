#!/usr/bin/env bash
# Diagnose why __half operator!= compiles on one box but not another.
# Run this on BOTH the local machine and the vast server, compare output.
set +e

echo "############ HALF-OPERATOR DIAGNOSTIC ############"
echo "== host =="; hostname 2>/dev/null; echo
echo "== nvcc version =="; nvcc --version 2>&1; echo
echo "== nvcc path =="; which nvcc; echo
echo "== cuda_fp16.h header(s) on disk =="
ls -la /usr/local/cuda*/include/cuda_fp16.h /usr/local/cuda*/targets/*/include/cuda_fp16.h /usr/include/cuda_fp16.h 2>/dev/null
echo
echo "== which cuda_fp16.h does -I/usr/include actually resolve to =="
echo '#include <cuda_fp16.h>' | nvcc -I/usr/include -std=c++17 -x cu -M - 2>/dev/null | tr ' ' '\n' | grep cuda_fp16.h
echo

WORK="$(mktemp -d)"
cat > "$WORK/probe.cu" <<'EOF'
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
// EXACT pattern from Hgemm_dispatcher.cu:77 inside a __host__ function.
__host__ void host_probe(const __half beta) {
    if (beta != __float2half(1.0f)) { printf("orig-op ran\n"); }        // fails on old CUDA
    if (__half2float(beta) != 1.0f) { printf("fix ran\n"); }            // always host-safe
}
int main() {
#ifdef CUDA_VERSION
    printf("CUDA_VERSION=%d\n", CUDA_VERSION);
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
    printf("__CUDA_NO_HALF_OPERATORS__ = defined\n");
#else
    printf("__CUDA_NO_HALF_OPERATORS__ = NOT defined\n");
#endif
    host_probe(__float2half(0.5f));
    return 0;
}
EOF

echo "== TEST 1: does 'beta != __float2half(1.0f)' compile in host code? (build-flag mimic) =="
nvcc -I/usr/include -std=c++17 -arch=sm_89 --expt-relaxed-constexpr "$WORK/probe.cu" -o "$WORK/probe" 2>&1 \
  && { echo ">>> RESULT: COMPILES (host operators available)"; "$WORK/probe"; } \
  || echo ">>> RESULT: FAILS (host operator!= NOT available -> this is your bug)"
echo

rm -rf "$WORK"
echo "############ END ############"
