# Overlap Bench

This benchmark measures Computation (GEMM) and Communication (RCCL AllReduce) execution times on AMD GPUs. It supports enabling/disabling computation and communication overlap and running compute and communication kernels separately.

## Arguments

```

usage: overlap_bench.py [-h] [--comp_only] [--comm_only] [--no_overlap] [--with_overlap] [-m MATRIXM] [-n MATRIXN] [-k MATRIXK] [-s MESSAGE_SIZE] [-i ITERATIONS] [-w WARMUP_ITERS] [-v] [-p] [--csv_path CSV_PATH]

A simple Compute (GEMM) and Communication (RCCL) overlap benchmark.

optional arguments:
  -h, --help            show this help message and exit
  --comp_only           Execute GEMM kernels only
  --comm_only           Perform AllReduce Communication only
  --no_overlap          Execute GEMM and AllReduce without overlap
  --with_overlap        Execute GEMM and AllReduce with overlap
  -m MATRIXM, --matrixm MATRIXM
                        Size of input GEMM matrix M
  -n MATRIXN, --matrixn MATRIXN
                        Size of input GEMM matrix N
  -k MATRIXK, --matrixk MATRIXK
                        Size of input GEMM matrix K
  -s MESSAGE_SIZE, --message_size MESSAGE_SIZE
                        Message size per rank - AllReduce communication
  -i ITERATIONS, --iterations ITERATIONS
                        Number of iterations
  -w WARMUP_ITERS, --warmup_iters WARMUP_ITERS
                        Number of warm-up iterations
  -v, --validate        Validate computation and communication output
  -p, --profile         Enable profiling using rocProf
  --csv_path CSV_PATH   CSV file path to store perf numbers.

  ```

By default, the benchmark reports all four execution times, i.e reporting time taken for running compute (GEMM) only, communication (AllReduce) only, compute + communication without overlap, compute + communication with overlap. Please use the options above to run a particular mode of execution. 

Setting --comp_only parameter executes only GEMM operation, whereas setting --comm_only executes only AllReduce operation. --no_overlap option disables computation and communication overlap, whereas --with_overlap enables computation and communication overlap. Please refer to the options above to set various parameters such as input matrix sizes, number of iterations, warm-up iterations, and enable profiling/validation.

## Installing Pytorch with ROCm support

Please follow these instructions to enable Pytorch support with ROCm using a docker image:
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#using-a-docker-image-with-pytorch-pre-installed


## Sample run command

GPU tested: MI300x (8 GPUs in a single node)

```
root@4b711f049a60:~/gemm-comm-overlap# torchrun --nnodes=1 --nproc_per_node=8 overlap_bench.py
Overlap Bench v1.0

Matrix m   Matrix n   Matrix k   Size (Bytes)  Datatype  Iterations  Comp Time (ms)  Comm Time (ms)  No overlap (ms)  With overlap (ms) 
8192       8192       8192       512M          Half      100         1.75            3.31            5.51             4.3               
Speedup = 1.28

Setting different parameters:
torchrun --nnodes=1 --nproc_per_node=8 overlap_bench.py -m 16384 -n 16384 -k 16384 -s 1G -i 10 -w 2 --csv_path="bench_perf.csv"
Overlap Bench v1.0
#
Matrix m   Matrix n   Matrix k   Size (Bytes)  Datatype  Iterations  Comp Time (ms)  Comm Time (ms)  No overlap (ms)  With overlap (ms) 
16384      16384      16384      1G            Half      10          14.30           6.48            20.41            18.06             
# Speedup = 1.13
# Data written to CSV file
```

## Enable profiling

```
When enabling profiling, please ensure that only one of the execution modes is selected. By default, the benchmark reports all four execution times.

<!-- start:code block -->
Enable Profiling with computation and communication overlap
rocprof --flush-rate 10ms --obj-tracking on --timestamp on --hip-trace --roctx-trace -d rocout torchrun --nnodes=1 --nproc_per_node=8 overlap_bench.py --with_overlap --profile

Enable profiling without overlap
rocprof --flush-rate 10ms --obj-tracking on --timestamp on --hip-trace --roctx-trace -d rocout torchrun --nnodes=1 --nproc_per_node=8 overlap_bench.py --no_overlap --profile
<!-- end:code block -->
```

To view the profiler traces, simply upload the results.json file in https://www.ui.perfetto.dev/

With overlap:
![image](https://github.com/user-attachments/assets/93357c9d-81a2-4086-b4c9-26307243e5a3)

Without overlap:
![image](https://github.com/user-attachments/assets/5bb90723-3a95-48c3-b883-8e4f93808df1)

### To-do
1. Add Validation support
