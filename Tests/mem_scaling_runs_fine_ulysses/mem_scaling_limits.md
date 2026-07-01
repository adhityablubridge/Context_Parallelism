# Scaling Limits (max T that ran OK / first T that OOM'd)

| impl | label | rotator | max_T_ok | first_T_oom |
|---|---|---|---|---|
| BluTrain | 124M | ulysses | 3712 | 3840 |
| BluTrain | 163M | ulysses | 3584 | 3712 |
| BluTrain | 25M | ulysses | 7040 | 7168 |
| BluTrain | 44M | ulysses | 6912 | 7040 |
| PyTorch | 124M | allgather | 3456 | 3584 |
| PyTorch | 124M | alltoall | 3456 | 3584 |
| PyTorch | 163M | allgather | 3072 | 3200 |
| PyTorch | 163M | alltoall | 3072 | 3200 |
| PyTorch | 25M | allgather | 5376 | 5504 |
| PyTorch | 25M | alltoall | 5504 | 5632 |
| PyTorch | 44M | allgather | 5376 | 5504 |
| PyTorch | 44M | alltoall | 5376 | 5504 |
