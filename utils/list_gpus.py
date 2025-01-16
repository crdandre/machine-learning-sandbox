# Method 1: Using torch.cuda
import torch

def list_gpus_torch():
    if not torch.cuda.is_available():
        print("No CUDA devices available")
        return
        
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} (Memory: {props.total_memory / 1024**3:.1f}GB)")

# Method 2: Using nvidia-smi through subprocess
import subprocess
import json

def list_gpus_nvidia():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,uuid', 
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        for line in output.strip().split('\n'):
            idx, name, memory, uuid = line.split(', ')
            print(f"GPU {idx}: {name} (Memory: {float(memory)/1024:.1f}GB)")
            print(f"UUID: {uuid}")
    except:
        print("Failed to run nvidia-smi")

# You can run either or both:
print("PyTorch method:")
list_gpus_torch()
print("\nNVIDIA-SMI method:")
list_gpus_nvidia()