import torch
import torch.distributed as dist
import argparse
import os
import csv
import re

def init_dist_setup():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def parse_msg_inp(msg_size):
    inp_msg = msg_size

    if msg_size.isdigit():
        return inp_msg, int(msg_size)
    
    match = re.match(r"^(\d+)([KMG])$", msg_size, re.IGNORECASE)
    if match:
        num, unit = match.groups()
        num = int(num)

        if unit.upper() == 'K':
            converted_size = num * 1024
        elif unit.upper() == 'M':
            converted_size = num * 1024 ** 2
        else:
            converted_size = num * 1024 ** 3
        return inp_msg, converted_size
    
    raise argparse.ArgumentTypeError(f"Invalid size. Input must be a number followed by 'K', 'M', or 'G'.")        

class Bench:
    def __init__(self, args):
        torch.cuda.manual_seed_all(42)
        self.m = args.matrixm
        self.n = args.matrixn
        self.k = args.matrixk
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.size = args.message_size[1]
        self.iters = args.iterations
        self.warmup_iters = args.warmup_iters
        self.enable_profile = args.profile
        self.csv_path = args.csv_path

    def comp_only(self):
        A = torch.randn(self.m, self.n, dtype=torch.float16, device=f'cuda:{self.gpu_id}')
        B = torch.randn(self.n, self.k, dtype=torch.float16, device=f'cuda:{self.gpu_id}')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self.enable_profile:
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                # Benchmark GEMM with profiler
                for i in range(self.iters + self.warmup_iters):
                    if i == self.warmup_iters:
                        start_event.record()
                    C = A @ B

        else:
            # Benchmark GEMM
            for i in range(self.iters + self.warmup_iters):
                if i == self.warmup_iters:
                    start_event.record()
                C = A @ B

        end_event.record()
        torch.cuda.synchronize(device=f'cuda:{self.gpu_id}')
        elapsed_time = start_event.elapsed_time(end_event) / self.iters
        elapsed_time_tensor = torch.tensor(elapsed_time, device=f'cuda:{self.gpu_id}')
        
        dist.all_reduce(elapsed_time_tensor, op=dist.ReduceOp.AVG)

        return elapsed_time_tensor.item()

    def comm_only(self):
        AR_size = self.size // 2
        AR_tensor = torch.randn(AR_size, dtype=torch.float16, device=f'cuda:{self.gpu_id}')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self.enable_profile:
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                # Benchmark AR with profiler
                for i in range(self.iters + self.warmup_iters):
                    AR_tensor_clone = AR_tensor.detach().clone()
                    if i == self.warmup_iters:
                        start_event.record()
                    dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM)

        else:
            # Benchmark AR
            for i in range(self.iters + self.warmup_iters):
                AR_tensor_clone = AR_tensor.detach().clone()
                if i == self.warmup_iters:
                    start_event.record()
                dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM)
            
        end_event.record()
        torch.cuda.synchronize(device=f'cuda:{self.gpu_id}')
        elapsed_time = start_event.elapsed_time(end_event) / self.iters
        elapsed_time_tensor = torch.tensor(elapsed_time, device=f'cuda:{self.gpu_id}')

        dist.all_reduce(elapsed_time_tensor, op=dist.ReduceOp.AVG)
        
        return elapsed_time_tensor.item()

    def comp_comm_no_overlap(self):
        AR_size = self.size // 2
        AR_tensor = torch.randn(AR_size, dtype=torch.float16, device=f'cuda:{self.gpu_id}')
        A = torch.randn(self.m, self.n, dtype=torch.float16, device=f'cuda:{self.gpu_id}')
        B = torch.randn(self.n, self.k, dtype=torch.float16, device=f'cuda:{self.gpu_id}')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self.enable_profile:
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                # Benchmark nonconcurrent GEMM + AR with profiler
                for i in range(self.iters + self.warmup_iters):
                    AR_tensor_clone = AR_tensor.detach().clone()
                    if i == self.warmup_iters:
                        start_event.record()
                    dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM) # Blocking call
                    C = A @ B          
        
        else:
            # Benchmark nonconcurrent GEMM + AR
            for i in range(self.iters + self.warmup_iters):
                AR_tensor_clone = AR_tensor.detach().clone()
                if i == self.warmup_iters:
                    start_event.record()
                dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM) # Blocking call
                C = A @ B
        
        end_event.record()
        torch.cuda.synchronize(device=f'cuda:{self.gpu_id}')
        elapsed_time = start_event.elapsed_time(end_event) / self.iters
        elapsed_time_tensor = torch.tensor(elapsed_time, device=f'cuda:{self.gpu_id}')

        dist.all_reduce(elapsed_time_tensor, op=dist.ReduceOp.AVG)
        
        return elapsed_time_tensor.item()

    def comp_comm_overlap(self):
        AR_size = self.size // 2
        AR_tensor = torch.randn(AR_size, dtype=torch.float16, device=f'cuda:{self.gpu_id}')
        A = torch.randn(self.m, self.n, dtype=torch.float16, device=f'cuda:{self.gpu_id}')
        B = torch.randn(self.n, self.k, dtype=torch.float16, device=f'cuda:{self.gpu_id}')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        if self.enable_profile:
            with torch.autograd.profiler.emit_nvtx(record_shapes=True):
                # Benchmark concurrent GEMM + AR
                for i in range(self.iters + self.warmup_iters):
                    AR_tensor_clone = AR_tensor.detach().clone()
                    if i == self.warmup_iters:
                        start_event.record()
                    handle = dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM, async_op=True) # Non-Blocking call
                    C = A @ B
                    handle.wait()
        
        else:
            # Benchmark concurrent GEMM + AR
            for i in range(self.iters + self.warmup_iters):
                AR_tensor_clone = AR_tensor.detach().clone()
                if i == self.warmup_iters:
                    start_event.record()
                handle = dist.all_reduce(AR_tensor_clone, op=dist.ReduceOp.SUM, async_op=True) # Non-Blocking call
                C = A @ B
                handle.wait()
        
        end_event.record()
        torch.cuda.synchronize(device=f'cuda:{self.gpu_id}')
        elapsed_time = start_event.elapsed_time(end_event) / self.iters
        elapsed_time_tensor = torch.tensor(elapsed_time, device=f'cuda:{self.gpu_id}')

        dist.all_reduce(elapsed_time_tensor, op=dist.ReduceOp.AVG)
        
        return elapsed_time_tensor.item()

def main(args):
    assert(torch.cuda.is_available())
    # Set blas library to hipblaslt
    #torch.backends.cuda.preferred_blas_library(backend="hipblaslt")
    init_dist_setup()
    benchTest = Bench(args)
    elapsed_times = ['NA', 'NA', 'NA', 'NA']

    if (not args.comp_only) and (not args.comm_only) and (not args.no_overlap) and (not args.with_overlap):
        args.comp_only, args.comm_only, args.no_overlap, args.with_overlap = True, True, True, True
    
    if args.comp_only:
        elapsed_times[0] = str(round(benchTest.comp_only(), 2)) 

    if args.comm_only:
        elapsed_times[1] = str(round(benchTest.comm_only(), 2))

    if args.no_overlap:
        elapsed_times[2] = str(round(benchTest.comp_comm_no_overlap(), 2))

    if args.with_overlap:
        elapsed_times[3] = str(round(benchTest.comp_comm_overlap(), 2))

    if int(os.environ["LOCAL_RANK"]) == 0:
        print("Overlap Bench v1.0\n#")
        headers = ['Matrix m', 'Matrix n', 'Matrix k', 'Size (Bytes)', 'Datatype', 'Iterations', 'Comp Time (ms)', 'Comm Time (ms)', 'No overlap (ms)', 'With overlap (ms)']
        data = [args.matrixm, args.matrixn, args.matrixk, args.message_size[0], 'Half', args.iterations, elapsed_times[0], elapsed_times[1], elapsed_times[2], elapsed_times[3]]
        header_format = "{:<10} {:<10} {:<10} {:<13} {:<9} {:<11} {:<15} {:<15} {:<16} {:<18}"
        data_format = "{:<10} {:<10} {:<10} {:<13} {:<9} {:<11} {:<15} {:<15} {:<16} {:<18}"

        print(header_format.format(*headers))
        print(data_format.format(*data))

        if elapsed_times[2] != 'NA' and elapsed_times[3] != 'NA':
            print("# Speedup = {:.2f}".format(float(elapsed_times[2])/float(elapsed_times[3])))
        
        if args.csv_path:
            csv_file = os.path.isfile(args.csv_path)

            with open(args.csv_path, mode='a', newline='') as f:
                csv_writer = csv.writer(f)

                if not csv_file:
                    csv_writer.writerow(headers)
                
                csv_writer.writerow(data)
            print('# Data written to CSV file')

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple Compute (GEMM) and Communication (RCCL) overlap benchmark.")
    parser.add_argument("--comp_only", action="store_true", help="Execute GEMM kernels only")
    parser.add_argument("--comm_only", action="store_true", help="Perform AllReduce Communication only")
    parser.add_argument("--no_overlap", action="store_true", help="Execute GEMM and AllReduce without overlap")
    parser.add_argument("--with_overlap", action="store_true", help="Execute GEMM and AllReduce with overlap")
    parser.add_argument("-m", "--matrixm", default=8192, type=int, help="Size of input GEMM matrix M")
    parser.add_argument("-n", "--matrixn", default=8192, type=int, help="Size of input GEMM matrix N")
    parser.add_argument("-k", "--matrixk", default=8192, type=int, help="Size of input GEMM matrix K")
    parser.add_argument("-s", "--message_size", default="512M", type=parse_msg_inp, help="Message size per rank - AllReduce communication")
    parser.add_argument("-i", "--iterations", default=100, type=int, help="Number of iterations")
    parser.add_argument("-w", "--warmup_iters", default=20, type=int, help="Number of warm-up iterations")
    parser.add_argument("-v", "--validate", action="store_true", help="Validate computation and communication output")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling using rocProf")
    parser.add_argument("--csv_path", type=str, help="CSV file path to store perf numbers.")
    
    args = parser.parse_args()
    
    main(args)
