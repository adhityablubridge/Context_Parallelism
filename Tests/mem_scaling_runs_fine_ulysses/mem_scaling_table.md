# Memory Scaling Results

`smi_max_used_mb` = max per-GPU MiB from live nvidia-smi (includes CUDA context/NCCL), filtered to the run's GPUs. `peak_mb` = each impl's native peak: PyTorch=torch max_reserved, C++=cudaMemGetInfo used (see `peak_src`). Blank rows = OOM (no snapshot produced).

| impl | label | params | rotator | n_embd | n_layer | n_head | weight_tying | T | world_size | smi_max_used_mb | peak_mb | peak_src | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BluTrain | 124M | 124475904 | ulysses | 768 | 12 | 12 | 1 | 1024 | 2 | 4997 | 4996.2 | cudaMemGetInfo | OK |
| BluTrain | 124M | 125262336 | ulysses | 768 | 12 | 12 | 1 | 2048 | 2 | 8389 | 8388.2 | cudaMemGetInfo | OK |
| BluTrain | 124M | 126048768 | ulysses | 768 | 12 | 12 | 1 | 3072 | 2 | 10181 | 10180.2 | cudaMemGetInfo | OK |
| BluTrain | 124M | 126441984 | ulysses | 768 | 12 | 12 | 1 | 3584 | 2 | 11173 | 11172.2 | cudaMemGetInfo | OK |
| BluTrain | 124M | 126540288 | ulysses | 768 | 12 | 12 | 1 | 3712 | 2 | 11653 | 11652.2 | cudaMemGetInfo | OK |
| BluTrain | 124M |  | ulysses | 768 | 12 | 12 | 1 | 3840 | 2 |  |  |  | OOM |
| BluTrain | 124M |  | ulysses | 768 | 12 | 12 | 1 | 4096 | 2 |  |  |  | OOM |
| BluTrain | 163M | 163109376 | ulysses | 768 | 12 | 12 | 0 | 1024 | 2 | 5445 | 5444.2 | cudaMemGetInfo | OK |
| BluTrain | 163M | 163895808 | ulysses | 768 | 12 | 12 | 0 | 2048 | 2 | 8869 | 8868.2 | cudaMemGetInfo | OK |
| BluTrain | 163M | 164682240 | ulysses | 768 | 12 | 12 | 0 | 3072 | 2 | 10565 | 10564.2 | cudaMemGetInfo | OK |
| BluTrain | 163M | 165075456 | ulysses | 768 | 12 | 12 | 0 | 3584 | 2 | 11621 | 11620.2 | cudaMemGetInfo | OK |
| BluTrain | 163M |  | ulysses | 768 | 12 | 12 | 0 | 3712 | 2 |  |  |  | OOM |
| BluTrain | 163M |  | ulysses | 768 | 12 | 12 | 0 | 3840 | 2 |  |  |  | OOM |
| BluTrain | 163M |  | ulysses | 768 | 12 | 12 | 0 | 4096 | 2 |  |  |  | OOM |
| BluTrain | 25M | 25034112 | ulysses | 384 | 3 | 6 | 1 | 1024 | 2 | 2181 | 2180.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 25427328 | ulysses | 384 | 3 | 6 | 1 | 2048 | 2 | 3717 | 3716.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 26213760 | ulysses | 384 | 3 | 6 | 1 | 4096 | 2 | 6981 | 6980.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 27000192 | ulysses | 384 | 3 | 6 | 1 | 6144 | 2 | 10053 | 10052.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 27196800 | ulysses | 384 | 3 | 6 | 1 | 6656 | 2 | 11109 | 11108.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 27295104 | ulysses | 384 | 3 | 6 | 1 | 6912 | 2 | 11557 | 11556.2 | cudaMemGetInfo | OK |
| BluTrain | 25M | 27344256 | ulysses | 384 | 3 | 6 | 1 | 7040 | 2 | 11781 | 11780.2 | cudaMemGetInfo | OK |
| BluTrain | 25M |  | ulysses | 384 | 3 | 6 | 1 | 7168 | 2 |  |  |  | OOM |
| BluTrain | 25M |  | ulysses | 384 | 3 | 6 | 1 | 8192 | 2 |  |  |  | OOM |
| BluTrain | 44M | 44350848 | ulysses | 384 | 3 | 6 | 0 | 1024 | 2 | 2629 | 2628.2 | cudaMemGetInfo | OK |
| BluTrain | 44M | 44744064 | ulysses | 384 | 3 | 6 | 0 | 2048 | 2 | 3813 | 3812.2 | cudaMemGetInfo | OK |
| BluTrain | 44M | 45530496 | ulysses | 384 | 3 | 6 | 0 | 4096 | 2 | 6981 | 6980.2 | cudaMemGetInfo | OK |
| BluTrain | 44M | 46316928 | ulysses | 384 | 3 | 6 | 0 | 6144 | 2 | 10085 | 10084.2 | cudaMemGetInfo | OK |
| BluTrain | 44M | 46513536 | ulysses | 384 | 3 | 6 | 0 | 6656 | 2 | 11461 | 11460.2 | cudaMemGetInfo | OK |
| BluTrain | 44M | 46611840 | ulysses | 384 | 3 | 6 | 0 | 6912 | 2 | 11813 | 11812.2 | cudaMemGetInfo | OK |
| BluTrain | 44M |  | ulysses | 384 | 3 | 6 | 0 | 7040 | 2 |  |  |  | OOM |
| BluTrain | 44M |  | ulysses | 384 | 3 | 6 | 0 | 7168 | 2 |  |  |  | OOM |
| BluTrain | 44M |  | ulysses | 384 | 3 | 6 | 0 | 8192 | 2 |  |  |  | OOM |
| PyTorch | 124M | 124475904 | allgather | 768 | 12 | 12 | True | 1024 | 2 | 4869 | 4604.0 | torch_reserved | OK |
| PyTorch | 124M | 125262336 | allgather | 768 | 12 | 12 | True | 2048 | 2 | 8061 | 7796.0 | torch_reserved | OK |
| PyTorch | 124M | 126048768 | allgather | 768 | 12 | 12 | True | 3072 | 2 | 10519 | 10254.0 | torch_reserved | OK |
| PyTorch | 124M | 126245376 | allgather | 768 | 12 | 12 | True | 3328 | 2 | 11361 | 11096.0 | torch_reserved | OK |
| PyTorch | 124M | 126343680 | allgather | 768 | 12 | 12 | True | 3456 | 2 | 11897 | 11632.0 | torch_reserved | OK |
| PyTorch | 124M |  | allgather | 768 | 12 | 12 | 1 | 3584 | 2 |  |  |  | OOM |
| PyTorch | 124M |  | allgather | 768 | 12 | 12 | 1 | 4096 | 2 |  |  |  | OOM |
| PyTorch | 124M | 124475904 | alltoall | 768 | 12 | 12 | True | 1024 | 2 | 4857 | 4592.0 | torch_reserved | OK |
| PyTorch | 124M | 125262336 | alltoall | 768 | 12 | 12 | True | 2048 | 2 | 8037 | 7772.0 | torch_reserved | OK |
| PyTorch | 124M | 126048768 | alltoall | 768 | 12 | 12 | True | 3072 | 2 | 10483 | 10218.0 | torch_reserved | OK |
| PyTorch | 124M | 126245376 | alltoall | 768 | 12 | 12 | True | 3328 | 2 | 11345 | 11080.0 | torch_reserved | OK |
| PyTorch | 124M | 126343680 | alltoall | 768 | 12 | 12 | True | 3456 | 2 | 11857 | 11592.0 | torch_reserved | OK |
| PyTorch | 124M |  | alltoall | 768 | 12 | 12 | 1 | 3584 | 2 |  |  |  | OOM |
| PyTorch | 124M |  | alltoall | 768 | 12 | 12 | 1 | 4096 | 2 |  |  |  | OOM |
| PyTorch | 163M | 163109376 | allgather | 768 | 12 | 12 | False | 1024 | 2 | 5313 | 5048.0 | torch_reserved | OK |
| PyTorch | 163M | 163895808 | allgather | 768 | 12 | 12 | False | 2048 | 2 | 8209 | 7944.0 | torch_reserved | OK |
| PyTorch | 163M | 164682240 | allgather | 768 | 12 | 12 | False | 3072 | 2 | 11847 | 11582.0 | torch_reserved | OK |
| PyTorch | 163M |  | allgather | 768 | 12 | 12 | 0 | 3200 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | allgather | 768 | 12 | 12 | 0 | 3328 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | allgather | 768 | 12 | 12 | 0 | 3584 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | allgather | 768 | 12 | 12 | 0 | 4096 | 2 |  |  |  | OOM |
| PyTorch | 163M | 163109376 | alltoall | 768 | 12 | 12 | False | 1024 | 2 | 5301 | 5036.0 | torch_reserved | OK |
| PyTorch | 163M | 163895808 | alltoall | 768 | 12 | 12 | False | 2048 | 2 | 8185 | 7920.0 | torch_reserved | OK |
| PyTorch | 163M | 164682240 | alltoall | 768 | 12 | 12 | False | 3072 | 2 | 11811 | 11546.0 | torch_reserved | OK |
| PyTorch | 163M |  | alltoall | 768 | 12 | 12 | 0 | 3200 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | alltoall | 768 | 12 | 12 | 0 | 3328 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | alltoall | 768 | 12 | 12 | 0 | 3584 | 2 |  |  |  | OOM |
| PyTorch | 163M |  | alltoall | 768 | 12 | 12 | 0 | 4096 | 2 |  |  |  | OOM |
| PyTorch | 25M | 25034112 | allgather | 384 | 3 | 6 | True | 1024 | 2 | 2541 | 2268.0 | torch_reserved | OK |
| PyTorch | 25M | 25427328 | allgather | 384 | 3 | 6 | True | 2048 | 2 | 4679 | 4414.0 | torch_reserved | OK |
| PyTorch | 25M | 26213760 | allgather | 384 | 3 | 6 | True | 4096 | 2 | 8979 | 8714.0 | torch_reserved | OK |
| PyTorch | 25M | 26606976 | allgather | 384 | 3 | 6 | True | 5120 | 2 | 11123 | 10858.0 | torch_reserved | OK |
| PyTorch | 25M | 26705280 | allgather | 384 | 3 | 6 | True | 5376 | 2 | 11655 | 11390.0 | torch_reserved | OK |
| PyTorch | 25M |  | allgather | 384 | 3 | 6 | 1 | 5504 | 2 |  |  |  | OOM |
| PyTorch | 25M |  | allgather | 384 | 3 | 6 | 1 | 5632 | 2 |  |  |  | OOM |
| PyTorch | 25M |  | allgather | 384 | 3 | 6 | 1 | 6144 | 2 |  |  |  | OOM |
| PyTorch | 25M |  | allgather | 384 | 3 | 6 | 1 | 8192 | 2 |  |  |  | OOM |
| PyTorch | 25M | 25034112 | alltoall | 384 | 3 | 6 | True | 1024 | 2 | 2537 | 2272.0 | torch_reserved | OK |
| PyTorch | 25M | 25427328 | alltoall | 384 | 3 | 6 | True | 2048 | 2 | 4689 | 4424.0 | torch_reserved | OK |
| PyTorch | 25M | 26213760 | alltoall | 384 | 3 | 6 | True | 4096 | 2 | 8955 | 8690.0 | torch_reserved | OK |
| PyTorch | 25M | 26606976 | alltoall | 384 | 3 | 6 | True | 5120 | 2 | 11093 | 10828.0 | torch_reserved | OK |
| PyTorch | 25M | 26705280 | alltoall | 384 | 3 | 6 | True | 5376 | 2 | 11623 | 11358.0 | torch_reserved | OK |
| PyTorch | 25M | 26754432 | alltoall | 384 | 3 | 6 | True | 5504 | 2 | 11893 | 11628.0 | torch_reserved | OK |
| PyTorch | 25M |  | alltoall | 384 | 3 | 6 | 1 | 5632 | 2 |  |  |  | OOM |
| PyTorch | 25M |  | alltoall | 384 | 3 | 6 | 1 | 6144 | 2 |  |  |  | OOM |
| PyTorch | 25M |  | alltoall | 384 | 3 | 6 | 1 | 8192 | 2 |  |  |  | OOM |
| PyTorch | 44M | 44350848 | allgather | 384 | 3 | 6 | False | 1024 | 2 | 3009 | 2736.0 | torch_reserved | OK |
| PyTorch | 44M | 44744064 | allgather | 384 | 3 | 6 | False | 2048 | 2 | 4753 | 4488.0 | torch_reserved | OK |
| PyTorch | 44M | 45530496 | allgather | 384 | 3 | 6 | False | 4096 | 2 | 9053 | 8788.0 | torch_reserved | OK |
| PyTorch | 44M | 45923712 | allgather | 384 | 3 | 6 | False | 5120 | 2 | 11197 | 10932.0 | torch_reserved | OK |
| PyTorch | 44M | 46022016 | allgather | 384 | 3 | 6 | False | 5376 | 2 | 11729 | 11464.0 | torch_reserved | OK |
| PyTorch | 44M |  | allgather | 384 | 3 | 6 | 0 | 5504 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | allgather | 384 | 3 | 6 | 0 | 5632 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | allgather | 384 | 3 | 6 | 0 | 6144 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | allgather | 384 | 3 | 6 | 0 | 8192 | 2 |  |  |  | OOM |
| PyTorch | 44M | 44350848 | alltoall | 384 | 3 | 6 | False | 1024 | 2 | 3005 | 2740.0 | torch_reserved | OK |
| PyTorch | 44M | 44744064 | alltoall | 384 | 3 | 6 | False | 2048 | 2 | 4763 | 4498.0 | torch_reserved | OK |
| PyTorch | 44M | 45530496 | alltoall | 384 | 3 | 6 | False | 4096 | 2 | 9029 | 8764.0 | torch_reserved | OK |
| PyTorch | 44M | 45923712 | alltoall | 384 | 3 | 6 | False | 5120 | 2 | 11167 | 10902.0 | torch_reserved | OK |
| PyTorch | 44M | 46022016 | alltoall | 384 | 3 | 6 | False | 5376 | 2 | 11697 | 11432.0 | torch_reserved | OK |
| PyTorch | 44M |  | alltoall | 384 | 3 | 6 | 0 | 5504 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | alltoall | 384 | 3 | 6 | 0 | 5632 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | alltoall | 384 | 3 | 6 | 0 | 6144 | 2 |  |  |  | OOM |
| PyTorch | 44M |  | alltoall | 384 | 3 | 6 | 0 | 8192 | 2 |  |  |  | OOM |
