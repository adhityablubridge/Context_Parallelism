blubridge@blubridge:/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch$ CUDA_VISIBLE_DEVICES=4,5 PT_TF32=1 torchrun --standalone --nnodes=1 --nproc-per-node=2 gpt2_cp_attnstyle_fp32.py
W0629 16:02:29.629000 3377111 torch/distributed/run.py:851]
W0629 16:02:29.629000 3377111 torch/distributed/run.py:851] *****************************************
W0629 16:02:29.629000 3377111 torch/distributed/run.py:851] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0629 16:02:29.629000 3377111 torch/distributed/run.py:851] *****************************************
Traceback (most recent call last):
  File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 41, in <module>
    from torch.distributed.tensor.experimental._attention import (
ImportError: cannot import name '_AttentionContextParallel' from 'torch.distributed.tensor.experimental._attention' (/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/tensor/experimental/_attention.py)
Traceback (most recent call last):
  File "/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism/Scripts/Pytorch/gpt2_cp_attnstyle_fp32.py", line 41, in <module>
    from torch.distributed.tensor.experimental._attention import (
ImportError: cannot import name '_AttentionContextParallel' from 'torch.distributed.tensor.experimental._attention' (/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/tensor/experimental/_attention.py)
E0629 16:02:34.714000 3377111 torch/distributed/elastic/multiprocessing/api.py:986] failed (exitcode: 1) local_rank: 0 (pid: 3377924) of binary: /usr/bin/python3
Traceback (most recent call last):
  File "/home/blubridge/.local/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
  File "/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 990, in main
    run(args)
  File "/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/run.py", line 981, in run
    elastic_launch(
  File "/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/blubridge/.local/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 317, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
gpt2_cp_attnstyle_fp32.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-06-29_16:02:34
  host      : blubridge
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 3377925)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-06-29_16:02:34
  host      : blubridge
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 3377924)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================