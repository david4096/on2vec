"""
Device utilities for GPU/CPU management in on2vec.

This module provides utilities for automatic device detection and tensor placement
to optimize performance on available hardware (CUDA, MPS, CPU).
"""

import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch computations.

    Args:
        device (str, optional): Specific device to use ('cpu', 'cuda', 'mps', 'auto')
                               If None or 'auto', automatically selects best available
        verbose (bool): Whether to log device selection information

    Returns:
        torch.device: The device to use for computations
    """
    if device is None:
        device = 'auto'

    if device == 'auto':
        # Automatic device selection based on availability
        if torch.cuda.is_available():
            selected_device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(0)
            if verbose:
                logger.info(f"🚀 Using CUDA GPU: {device_name}")
                logger.info(f"   CUDA version: {torch.version.cuda}")
                logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            selected_device = torch.device('mps')
            if verbose:
                logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
        else:
            selected_device = torch.device('cpu')
            if verbose:
                logger.info("💻 Using CPU (no GPU acceleration available)")
    else:
        # Explicit device selection
        if device == 'cuda':
            if torch.cuda.is_available():
                selected_device = torch.device('cuda')
                if verbose:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"🚀 Using CUDA GPU (explicit): {device_name}")
            else:
                logger.warning("❌ CUDA requested but not available, falling back to CPU")
                selected_device = torch.device('cpu')
        elif device == 'mps':
            if torch.backends.mps.is_available():
                selected_device = torch.device('mps')
                if verbose:
                    logger.info("🍎 Using MPS (explicit)")
            else:
                logger.warning("❌ MPS requested but not available, falling back to CPU")
                selected_device = torch.device('cpu')
        else:
            selected_device = torch.device(device)
            if verbose:
                logger.info(f"💻 Using device (explicit): {device}")

    return selected_device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Information about available compute devices
    """
    info = {
        'cpu': True,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        'recommended_device': get_device().type
    }

    if info['cuda_available']:
        info['cuda_devices'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


def move_to_device(data, device: torch.device, non_blocking: bool = True):
    """
    Move data (tensors, models, or nested structures) to specified device.

    Args:
        data: Data to move (tensor, model, dict, list, tuple)
        device: Target device
        non_blocking: Use non-blocking transfers when possible

    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, torch.nn.Module):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device, non_blocking) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        moved_data = [move_to_device(item, device, non_blocking) for item in data]
        return type(data)(moved_data)  # Preserve original type (list/tuple)
    else:
        # Return data unchanged if it's not a tensor or module
        return data


def get_memory_usage(device: torch.device) -> Tuple[float, float]:
    """
    Get memory usage for the specified device.

    Args:
        device: Device to check memory for

    Returns:
        tuple: (used_memory_gb, total_memory_gb) or (0, 0) if not applicable
    """
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1e9
        cached = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        return cached, total
    else:
        # CPU/MPS memory tracking not directly available through PyTorch
        return 0.0, 0.0


def optimize_for_device(device: torch.device) -> dict:
    """
    Get optimal settings for the specified device.

    Args:
        device: Target device

    Returns:
        dict: Optimization settings for the device
    """
    settings = {
        'pin_memory': False,
        'non_blocking': False,
        'autocast_enabled': False,
        'autocast_dtype': torch.float16
    }

    if device.type == 'cuda':
        settings.update({
            'pin_memory': True,
            'non_blocking': True,
            'autocast_enabled': True,
            'autocast_dtype': torch.float16
        })
    elif device.type == 'mps':
        settings.update({
            'pin_memory': False,  # MPS doesn't support pinned memory
            'non_blocking': False,
            'autocast_enabled': False,  # MPS autocast is experimental
        })

    return settings


def log_device_performance_tips(device: torch.device):
    """
    Log performance optimization tips for the selected device.

    Args:
        device: The device being used
    """
    if device.type == 'cuda':
        logger.info("💡 GPU Performance Tips:")
        logger.info("   - Using mixed precision training (autocast) for faster training")
        logger.info("   - Pin memory enabled for faster data transfers")
        logger.info("   - Consider increasing batch size if memory allows")
    elif device.type == 'mps':
        logger.info("💡 MPS Performance Tips:")
        logger.info("   - MPS works best with float32 precision")
        logger.info("   - Some operations may fall back to CPU automatically")
        logger.info("   - Model size limited by unified memory architecture")
    else:
        logger.info("💡 CPU Performance Tips:")
        logger.info("   - Consider using smaller models or fewer epochs")
        logger.info("   - Enable CPU optimizations: export MKL_NUM_THREADS=1")
        logger.info("   - Use torch.set_num_threads() to control parallelism")