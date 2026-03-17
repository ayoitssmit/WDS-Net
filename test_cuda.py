import torch

def test_cuda():
    print("--- CUDA Availability Test ---")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Version Built With: {torch.version.cuda}")
    else:
        print("CUDA is NOT available. This usually means the CPU-only version of PyTorch is installed or GPU drivers are missing.")
        
if __name__ == "__main__":
    test_cuda()
