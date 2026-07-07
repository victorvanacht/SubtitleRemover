import torch

def check_pytorch():
    """Check if PyTorch is working and GPU is accessible."""
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get GPU count
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        # Get current GPU device
        current_gpu = torch.cuda.current_device()
        print(f"Current GPU device: {current_gpu}")
        
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(current_gpu)
        print(f"GPU name: {gpu_name}")
        
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # Test GPU computation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("GPU computation test: PASSED")
        except Exception as e:
            print(f"GPU computation test: FAILED - {e}")
    else:
        print("No GPU detected. PyTorch will use CPU.")
        print("To use GPU, ensure NVIDIA CUDA toolkit is installed.")

if __name__ == "__main__":
    check_pytorch()
