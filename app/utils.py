
from llama_cpp import Llama
import os
import psutil

def detect_hardware_and_load():
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpus = os.cpu_count() or 4 #counting threads
    
    if ram_gb < 8:
        repo = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
        file = "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
        context = 1024
    elif ram_gb < 16:
        repo = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
        file = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        context = 2048
    else:
        repo = "bartowski/Qwen2.5-7B-Instruct-GGUF"
        file = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        context = 4096


    threads = max(1, cpus - 1) #we use all the threads except for one

    return Llama.from_pretrained(
        repo_id=repo,
        filename=file,
        n_ctx=context,
        n_threads=threads
    )