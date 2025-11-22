# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Device utilities for cross-platform GPU support (CUDA, MPS, CPU)."""

import torch
import warnings
from typing import Optional, Union, Dict


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get the best available device or validate a specified device.
    
    Priority: MPS > CUDA > CPU
    
    Args:
        device: Optional device specification. If None, returns best available.
    
    Returns:
        torch.device: The selected device.
    """
    if device is not None:
        return torch.device(device)
    
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            # Test MPS availability by creating a small tensor
            _ = torch.zeros(1).to("mps")
            return torch.device("mps")
        except Exception as e:
            warnings.warn(f"MPS is available but failed to initialize: {e}. Falling back to CPU.")
    
    # Check for CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Fallback to CPU
    return torch.device("cpu")


def is_gpu_available() -> bool:
    """
    Check if any GPU (CUDA or MPS) is available.
    
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    if torch.backends.mps.is_available():
        return True
    if torch.cuda.is_available():
        return True
    return False


def get_device_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get memory information for the specified device.
    
    Args:
        device: Device to query. If None, uses current device.
    
    Returns:
        Dict with memory info in MB:
            - allocated: Currently allocated memory
            - reserved: Reserved memory
            - max_allocated: Maximum allocated over time
            - max_reserved: Maximum reserved over time
    """
    if device is None:
        device = get_device()
    
    device_str = str(device)
    
    if "cuda" in device_str:
        return {
            "allocated": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved": torch.cuda.memory_reserved(device) / 1024**2,
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**2,
            "max_reserved": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    elif "mps" in device_str:
        # MPS memory tracking is limited in PyTorch
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**2
            return {
                "allocated": allocated,
                "reserved": allocated,  # MPS doesn't separate allocated/reserved
                "max_allocated": allocated,
                "max_reserved": allocated,
            }
        except Exception:
            return {
                "allocated": 0.0,
                "reserved": 0.0,
                "max_allocated": 0.0,
                "max_reserved": 0.0,
            }
    else:
        # CPU doesn't track memory this way
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "max_allocated": 0.0,
            "max_reserved": 0.0,
        }


def empty_cache(device: Optional[torch.device] = None):
    """
    Empty the cache for the specified device.
    
    Args:
        device: Device to clear cache for. If None, uses current device.
    """
    if device is None:
        device = get_device()
    
    device_str = str(device)
    
    if "cuda" in device_str:
        torch.cuda.empty_cache()
    elif "mps" in device_str:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass  # MPS cache clearing may not be available in all PyTorch versions


def set_device(device: Union[int, str, torch.device]):
    """
    Set the current device.
    
    Args:
        device: Device to set as current.
    """
    if isinstance(device, int):
        # Assume it's a CUDA device index
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        else:
            warnings.warn(f"CUDA not available, cannot set device {device}")
    else:
        device_obj = torch.device(device) if isinstance(device, str) else device
        device_str = str(device_obj)
        
        if "cuda" in device_str:
            if torch.cuda.is_available():
                # Extract device index if specified
                if ":" in device_str:
                    device_idx = int(device_str.split(":")[1])
                    torch.cuda.set_device(device_idx)
                else:
                    torch.cuda.set_device(0)
        # MPS doesn't have a set_device equivalent


def to_device(
    tensor_or_model: Union[torch.Tensor, torch.nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = False,
) -> Union[torch.Tensor, torch.nn.Module]:
    """
    Move tensor or model to the specified device.
    
    Args:
        tensor_or_model: Tensor or model to move.
        device: Target device. If None, uses best available.
        non_blocking: Whether to use non-blocking transfer.
    
    Returns:
        Tensor or model on the target device.
    """
    if device is None:
        device = get_device()
    
    # MPS doesn't support non_blocking transfers as of some PyTorch versions
    device_str = str(device)
    if "mps" in device_str:
        non_blocking = False
    
    return tensor_or_model.to(device, non_blocking=non_blocking)


def get_device_name(device: Optional[torch.device] = None) -> str:
    """
    Get a human-readable name for the device.
    
    Args:
        device: Device to get name for. If None, uses current device.
    
    Returns:
        str: Human-readable device name.
    """
    if device is None:
        device = get_device()
    
    device_str = str(device)
    
    if "cuda" in device_str:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            return f"{props.name} (CUDA {props.major}.{props.minor})"
        return "CUDA (not available)"
    elif "mps" in device_str:
        return "Apple Metal Performance Shaders (MPS)"
    else:
        return "CPU"


def get_device_info() -> str:
    """
    Get comprehensive device information as a formatted string.
    
    Returns:
        str: Formatted device information.
    """
    device = get_device()
    device_name = get_device_name(device)
    mem_info = get_device_memory_info(device)
    
    info = [
        f"Device: {device_name}",
        f"PyTorch version: {torch.__version__}",
    ]
    
    if "cuda" in str(device):
        info.append(f"CUDA arch: {torch.cuda.get_arch_list()}")
        info.append(
            f"Memory: {mem_info['allocated']:.1f} MiB allocated, "
            f"{mem_info['reserved']:.1f} MiB reserved"
        )
    elif "mps" in str(device):
        if mem_info['allocated'] > 0:
            info.append(f"Memory: {mem_info['allocated']:.1f} MiB allocated")
        else:
            info.append("Memory tracking: Not available")
    
    return "\n".join(info)
