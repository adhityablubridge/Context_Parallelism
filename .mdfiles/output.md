(adhitya_venv) blubridge@blubridge:/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism$ # 1. confirm it IS the import error
cat Tests/mem_scaling_runs_fine/logs/PT_25M_alltoall_T1024_ws2.log

# 2. torch version
python3 -c "import torch; print(torch.__version__)"

# 3. what the CP attention module exposes in this torch
python3 -c "import torch.distributed.tensor.experimental._attention as a; print(sorted(n for n in dir(a) if not n.startswith('__')))"
W0629 16:48:34.764000 3838429 torch/distributed/run.py:803]
W0629 16:48:34.764000 3838429 torch/distributed/run.py:803] *****************************************
W0629 16:48:34.764000 3838429 torch/distributed/run.py:803] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0629 16:48:34.764000 3838429 torch/distributed/run.py:803] *****************************************
/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision= 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
[PT_TF32] allow_tf32=True (matching C++ TF32 matmul precision)
/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/backends/__init__.py:46: UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision= 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9. Please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:80.)
  self.setter(val)
=== GPT-2 CP via _AttentionContextParallel (FP32, HeadTail LB) ===
  Applied _AttentionContextParallel to 3 blocks
Configuration:
  vocab_size:     50304
  context_length: 1024
  n_embd:         384
  n_layers:       3
  n_heads:        6
  B=4, T=1024
  cp_world_size:  2
  global_batch:   4096
  grad_accum_steps: 1
  Parameters:          25034112
  max_steps:      2
  warmup_steps:   1
  CP API:         _AttentionContextParallel (ParallelStyle)
[rank0]: Traceback (most recent call last):
[rank0]:   File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 780, in <module>
[rank0]:     train_loader = DataLoaderLite(B, T, "train")
[rank0]:   File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 546, in __init__
[rank0]:     for s in os.listdir(data_root)
[rank0]: FileNotFoundError: [Errno 2] No such file or directory: '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data'

[INSTRUMENTATION rank=0] patched: {'bwd': True}
[INSTRUMENTATION rank=0] CP backward called 0 times
[rank1]: Traceback (most recent call last):
[rank1]:   File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 780, in <module>
[rank1]:     train_loader = DataLoaderLite(B, T, "train")
[rank1]:   File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 546, in __init__
[rank1]:     for s in os.listdir(data_root)
[rank1]: FileNotFoundError: [Errno 2] No such file or directory: '/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data'

[INSTRUMENTATION rank=1] patched: {'bwd': True}
[INSTRUMENTATION rank=1] CP backward called 0 times
[rank0]:[W629 16:48:47.891352635 ProcessGroupNCCL.cpp:1524] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
W0629 16:48:48.079000 3838429 torch/distributed/elastic/multiprocessing/api.py:908] Sending process 3839288 closing signal SIGTERM
E0629 16:48:48.500000 3838429 torch/distributed/elastic/multiprocessing/api.py:882] failed (exitcode: 1) local_rank: 0 (pid: 3839287) of binary: /mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/bin/python3
Traceback (most recent call last):
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 357, in wrapper
    return f(*args, **kwargs)
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/distributed/run.py", line 936, in main
    run(args)
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/distributed/run.py", line 927, in run
    elastic_launch(
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 156, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/mnt/volgrp03/3rd_floor/Adhitya/TensorParallelismBeta/DTensor/Pytorch/adhitya_venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 293, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-06-29_16:48:48
  host      : blubridge
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 3839287)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
2.9.0+cu128
['ABC', 'Any', 'BlockMask', 'Callable', 'DTensor', 'DeviceMesh', 'Enum', 'F', 'Generator', 'Optional', 'ParallelStyle', 'Protocol', 'Replicate', 'Shard', 'TorchFunctionMode', 'Union', '_AllGatherRotater', '_AllToAllRotater', '_AttentionContextParallel', '_AttentionOp', '_CausalBehavior', '_ContextParallelGlobalVars', '_ContextParallelOptions', '_DispatchMode', '_RingRotater', '_RotateMethod', '_SDPAMerger', '_context_parallel', '_context_parallel_buffers', '_cp_global_vars', '_cp_options', '_create_rotater', '_dispatch_mode', '_distribute_function', '_enable_cp_dispatcher', '_generate_round_robin_indices', '_is_causal_behavior', '_mask_mod_signature', '_maybe_wait', '_partial_update', '_replaced_functions', '_restore_function', '_scaled_dot_product_ring_cudnn_attention', '_scaled_dot_product_ring_cudnn_attention_backward', '_scaled_dot_product_ring_efficient_attention', '_scaled_dot_product_ring_efficient_attention_backward', '_scaled_dot_product_ring_flash_attention', '_scaled_dot_product_ring_flash_attention_backward', '_sdpa_handler', '_set_cp_global_var', '_templated_ring_attention', '_templated_ring_attention_backward', 'abstractmethod', 'aten', 'auto', 'context_parallel', 'context_parallel_unshard', 'contextlib', 'create_block_mask', 'create_cp_block_mask', 'customized_ops', 'dataclass','dist', 'distribute_module', 'distribute_tensor', 'ft_c', 'itertools', 'logger', 'logging', 'nn', 'set_rotate_method', 'torch', 'types', 'weakref']