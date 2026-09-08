import sys

# Safely check loaded modules without an explicit 'import torch' in your code
if "torch" in sys.modules or __import__("torch"):
    import torch
    print("--- CUDA STATUS CHECK ---")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
