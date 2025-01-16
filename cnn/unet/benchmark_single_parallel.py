import subprocess
import time
from datetime import datetime
import json

def run_benchmark(num_epochs=1, batch_size=16, max_images=1000):
    results = {}
    
    # Single GPU benchmark
    print("\n=== Running Single GPU Benchmark ===")
    start_time = time.time()
    subprocess.run([
        "python", 
        "train_single.py",
        f"--epochs={num_epochs}",
        f"--batch-size={batch_size}",
        f"--max-images={max_images}"
    ], check=True)
    single_gpu_time = time.time() - start_time
    results['single_gpu'] = single_gpu_time
    
    # Multi GPU benchmark
    print("\n=== Running Multi GPU Benchmark ===")
    start_time = time.time()
    subprocess.run([
        "torchrun",
        "--nproc_per_node=2",
        "train_parallel.py",
        f"--epochs={num_epochs}",
        f"--batch-size={batch_size}",
        f"--max-images={max_images}"
    ], check=True)
    multi_gpu_time = time.time() - start_time
    results['multi_gpu'] = multi_gpu_time
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results['speedup'] = single_gpu_time / multi_gpu_time
    results['config'] = {
        'epochs': num_epochs,
        'batch_size': batch_size
    }
    
    with open(f'benchmark_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n=== Benchmark Results ===")
    print(f"Single GPU Time: {single_gpu_time:.2f}s")
    print(f"Multi GPU Time: {multi_gpu_time:.2f}s")
    print(f"Speedup: {results['speedup']:.2f}x")

if __name__ == "__main__":
    run_benchmark(num_epochs=1, batch_size=16, max_images=1000)
