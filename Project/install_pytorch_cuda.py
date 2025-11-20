"""
Script to install PyTorch with CUDA support for GPU acceleration.
This will check your current installation and upgrade to CUDA-enabled PyTorch if needed.
"""

import subprocess
import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def run_command(command, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_pytorch():
    """Check current PyTorch installation."""
    print("=" * 60)
    print("Checking current PyTorch installation...")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch is installed: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[OK] CUDA is available: {cuda_version}")
            print(f"[OK] GPU detected: {gpu_name}")
            print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            return True, True  # Installed, CUDA available
        else:
            print("[WARNING] PyTorch is installed but CUDA is NOT available")
            if "+cpu" in torch.__version__:
                print("[WARNING] CPU-only version detected - needs upgrade")
            return True, False  # Installed, but no CUDA
    except ImportError:
        print("[ERROR] PyTorch is not installed")
        return False, False  # Not installed

def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    print("\n" + "=" * 60)
    print("Checking NVIDIA driver...")
    print("=" * 60)
    
    success, output, error = run_command("nvidia-smi", check=False)
    if success:
        print("[OK] NVIDIA driver is installed")
        # Extract CUDA version from nvidia-smi output
        lines = output.split('\n')
        for line in lines:
            if 'CUDA Version' in line:
                print(f"   {line.strip()}")
        return True
    else:
        print("[WARNING] NVIDIA driver not found or nvidia-smi not available")
        print("   Make sure you have NVIDIA drivers installed")
        return False

def install_pytorch_cuda(cuda_version="12.1"):
    """Install PyTorch with CUDA support."""
    print("\n" + "=" * 60)
    print(f"Installing PyTorch with CUDA {cuda_version} support...")
    print("=" * 60)
    
    # Uninstall existing PyTorch
    print("\n[INFO] Uninstalling existing PyTorch packages...")
    packages = ["torch", "torchvision", "torchaudio"]
    for package in packages:
        success, output, error = run_command(f"{sys.executable} -m pip uninstall {package} -y", check=False)
        if success:
            print(f"   [OK] Uninstalled {package}")
    
    # Install CUDA-enabled PyTorch
    print(f"\n[INFO] Installing PyTorch with CUDA {cuda_version}...")
    
    if cuda_version == "12.1":
        install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif cuda_version == "11.8":
        install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        print(f"[ERROR] Unsupported CUDA version: {cuda_version}")
        return False
    
    print(f"   Running: {install_cmd}")
    success, output, error = run_command(install_cmd, check=False)
    
    if success:
        print("[OK] Installation completed!")
        return True
    else:
        print(f"[ERROR] Installation failed:")
        print(error)
        return False

def verify_installation():
    """Verify PyTorch installation with CUDA."""
    print("\n" + "=" * 60)
    print("Verifying installation...")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"[OK] CUDA version: {cuda_version}")
            print(f"[OK] GPU: {gpu_name}")
            print(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            
            # Test a simple CUDA operation
            print("\n[TEST] Testing CUDA with a simple operation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("[OK] CUDA test passed! GPU is working correctly.")
            return True
        else:
            print("[ERROR] CUDA is still not available after installation")
            print("   This might be a driver or compatibility issue")
            return False
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False

def main():
    """Main installation process."""
    print("\n" + "=" * 60)
    print("PyTorch CUDA Installation Script")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Check your current PyTorch installation")
    print("2. Check NVIDIA driver")
    print("3. Install PyTorch with CUDA support if needed")
    print("4. Verify the installation")
    print("\n" + "=" * 60)
    
    # Check NVIDIA driver first
    driver_available = check_nvidia_driver()
    if not driver_available:
        print("\n[WARNING] NVIDIA driver not detected.")
        print("   Please install NVIDIA drivers first before proceeding.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Check current PyTorch
    pytorch_installed, cuda_available = check_pytorch()
    
    if pytorch_installed and cuda_available:
        print("\n[OK] PyTorch with CUDA is already installed and working!")
        print("   No action needed.")
        return
    
    # Ask user which CUDA version to install
    print("\n" + "=" * 60)
    print("CUDA Version Selection")
    print("=" * 60)
    print("RTX 4060 supports CUDA 11.8 and 12.1")
    print("Recommended: CUDA 12.1 (newer, better performance)")
    print("\n1. CUDA 12.1 (Recommended for RTX 4060)")
    print("2. CUDA 11.8 (Alternative)")
    
    choice = input("\nSelect CUDA version (1 or 2, default=1): ").strip()
    cuda_version = "12.1" if choice != "2" else "11.8"
    
    # Confirm installation
    print(f"\n[WARNING] This will uninstall your current PyTorch and install CUDA {cuda_version} version.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    # Install PyTorch with CUDA
    if install_pytorch_cuda(cuda_version):
        # Verify installation
        if verify_installation():
            print("\n" + "=" * 60)
            print("[SUCCESS] PyTorch with CUDA is now installed and working!")
            print("=" * 60)
            print("\nYou can now run the Streamlit app and it will use your RTX 4060 GPU!")
            print("Expected performance improvement: 3-5x faster processing!")
        else:
            print("\n[WARNING] Installation completed but verification failed.")
            print("   Please check your NVIDIA drivers and CUDA installation.")
    else:
        print("\n[ERROR] Installation failed. Please check the error messages above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

