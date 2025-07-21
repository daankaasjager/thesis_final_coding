import torch

def get_torch_dtype(precision):
    """Converts precision string from config to PyTorch dtype."""
    mapping = {
        "16-mixed": torch.float16,
        "bf16": torch.bfloat16,
        "32": torch.float32,  # Default to full precision
    }
    return mapping.get(precision, torch.float32)  # Defaults to float32 if unknown
