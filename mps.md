# MPS and CPU Device Support in ComfyUI Nodes

## Overview

This document explains how to implement **Apple Metal Performance Shaders (MPS)** and **CPU fallback** support in ComfyUI custom nodes. This is a **frontend toggle** feature that allows users to explicitly select their preferred device, with automatic detection as the default.

## What Is It?

**It's a UI toggle, not just a backend setting.**

Users see a dropdown in the ComfyUI node interface that lets them choose:
- `auto` - Automatically detects the best available device
- `cuda` - Force NVIDIA GPU
- `mps` - Force Apple Silicon GPU (Mac)
- `cpu` - Force CPU execution

### UI Example from Qwen-TTS Node

```python
"device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"})
```

This renders in ComfyUI as:
```
┌─────────────────────────────────────┐
│  Device:  [auto ▼]                  │
│           [cuda ]                   │
│           [mps  ]                   │
│           [cpu  ]                   │
└─────────────────────────────────────┘
```

## Why This Matters

### 1. **Apple Silicon Users**
- Macs with M1/M2/M3 chips can use MPS for GPU acceleration
- MPS performs poorly with `float32` - must use `float16` or `bfloat16`
- Some operations may not be supported on MPS and need CPU fallback

### 2. **CPU-Only Users**
- Users without GPUs can force CPU mode
- Useful for testing or when VRAM is exhausted
- Slower but ensures compatibility

### 3. **Debugging**
- Force specific devices to isolate issues
- Test model compatibility across platforms

## Implementation Guide

### Step 1: Add Device Toggle to Node UI

In your node's `INPUT_TYPES` class method:

```python
from typing import Dict, Any

class YourNode:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                # ... other inputs ...
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
            }
        }
```

### Step 2: Device Detection Utility Functions

Create reusable utility functions (based on VibeVoice-ComfyUI implementation):

```python
import torch
import logging

logger = logging.getLogger(__name__)

def get_optimal_device() -> str:
    """
    Get the best available device (cuda, mps, or cpu).
    
    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        String device identifier: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return "mps"
    else:
        logger.info("Using CPU device")
        return "cpu"

def get_device_for_loading(device: str) -> str:
    """
    Get device map string for model loading.
    
    Args:
        device: User-selected device ("auto", "cuda", "mps", or "cpu")
    
    Returns:
        Device string for transformers device_map parameter
    """
    if device == "auto":
        return get_optimal_device()
    return device

def get_optimal_dtype(device: str, precision: str = "bf16") -> torch.dtype:
    """
    Get optimal dtype for device.
    
    MPS performs poorly with float32, so we force float16 on Mac.
    
    Args:
        device: Device string ("cuda", "mps", "cpu")
        precision: User preference ("bf16" or "fp32")
    
    Returns:
        PyTorch dtype
    """
    if device == "mps":
        # MPS is slow with float32, use float16 or bfloat16
        if precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float16
    elif device == "cuda":
        return torch.bfloat16 if precision == "bf16" else torch.float32
    else:
        # CPU - use float32 for compatibility
        return torch.float32
```

### Step 3: Model Loading Function

```python
def load_model_with_device(
    model_path: str,
    device: str = "auto",
    precision: str = "bf16"
):
    """
    Load model with proper device and dtype configuration.
    
    Args:
        model_path: Path to model or HuggingFace repo ID
        device: Device selection ("auto", "cuda", "mps", "cpu")
        precision: Precision mode ("bf16" or "fp32")
    
    Returns:
        Loaded model
    """
    # Resolve device
    if device == "auto":
        device = get_optimal_device()
    
    # Get appropriate dtype
    dtype = get_optimal_dtype(device, precision)
    
    logger.info(f"Loading model on {device} with dtype {dtype}")
    
    # Load model with device_map
    # This automatically places model components on the specified device
    from transformers import AutoModel
    
    model = AutoModel.from_pretrained(
        model_path,
        device_map=device,  # "cuda", "mps", or "cpu"
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    return model
```

### Step 4: Node Implementation

```python
import torch
import numpy as np
from typing import Dict, Any, Tuple

class VoiceGenerationNode:
    """
    Example node with MPS/CPU support.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world"}),
                "model_name": (["model-v1", "model-v2"], {"default": "model-v1"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Audio/AI"
    
    def __init__(self):
        self.model = None
        self.current_device = None
    
    def generate(
        self,
        text: str,
        model_name: str,
        device: str,
        precision: str,
        seed: int = 0
    ) -> Tuple[Dict[str, Any]]:
        """
        Generate audio with device-specific optimizations.
        """
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Resolve device selection
        actual_device = device if device != "auto" else get_optimal_device()
        
        # MPS-specific: Set PyTorch backend
        if actual_device == "mps":
            # Ensure MPS is available
            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                print(f"⚠️ MPS not available, falling back to CPU")
                actual_device = "cpu"
        
        # Load model (with caching based on device)
        if self.model is None or self.current_device != actual_device:
            self.model = load_model_with_device(
                f"models/{model_name}",
                device=actual_device,
                precision=precision
            )
            self.current_device = actual_device
        
        # Move inputs to device
        inputs = self.prepare_inputs(text, device=actual_device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        # Process outputs
        audio = outputs.audio.detach().cpu().numpy()
        
        return ({
            "waveform": audio,
            "sample_rate": 24000
        },)
    
    def prepare_inputs(self, text: str, device: str) -> Dict[str, torch.Tensor]:
        """Prepare and move inputs to correct device."""
        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(device) for k, v in tokens.items()}
        
        return inputs
```

## Key Implementation Details

### 1. **MPS Detection Requires `hasattr()`**

Always check MPS availability with `hasattr()` because `torch.backends.mps` doesn't exist on non-Mac systems:

```python
# ✅ CORRECT
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    return "mps"

# ❌ WRONG - Will crash on non-Mac
if torch.backends.mps.is_available():
    return "mps"
```

### 2. **MPS Requires Float16**

MPS is extremely slow with `float32`. Always use `float16` or `bfloat16`:

```python
if device == "mps":
    dtype = torch.float16  # or torch.bfloat16
```

### 3. **Device Map for Transformers**

When using HuggingFace Transformers, use `device_map`:

```python
model = AutoModel.from_pretrained(
    model_path,
    device_map="mps",  # Automatically handles device placement
    torch_dtype=torch.float16
)
```

### 4. **Moving Tensors**

Always move tensors to the correct device before inference:

```python
# Move inputs
inputs = {k: v.to(device) for k, v in inputs.items()}

# Or for single tensor
input_ids = input_ids.to(device)
```

### 5. **Output Always to CPU**

Convert outputs to CPU before returning (for ComfyUI compatibility):

```python
output = model.generate(inputs)
audio = output.detach().cpu().numpy()
```

## Common Issues and Solutions

### Issue 1: MPS Not Available Error

```python
# Problem: torch.backends.mps doesn't exist on Linux/Windows
if torch.backends.mps.is_available():  # ❌ AttributeError

# Solution: Check with hasattr
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # ✅
```

### Issue 2: MPS Performance is Terrible

```python
# Problem: Using float32 on MPS
model = model.to("mps")  # Default dtype might be float32

# Solution: Force float16/bfloat16
dtype = torch.float16 if device == "mps" else torch.float32
model = model.to(device=device, dtype=dtype)
```

### Issue 3: Model Caching by Device

```python
class YourNode:
    def __init__(self):
        self.models = {}  # Cache per device
    
    def get_model(self, device: str):
        if device not in self.models:
            self.models[device] = load_model(device=device)
        return self.models[device]
```

### Issue 4: Operations Not Supported on MPS

Some PyTorch operations don't work on MPS. Handle with fallback:

```python
try:
    result = some_operation(tensor)
except RuntimeError as e:
    if "MPS" in str(e):
        # Fallback to CPU
        result = some_operation(tensor.cpu()).to(tensor.device)
    else:
        raise
```

## Testing Your Implementation

### Test 1: Device Detection

```python
# Test auto-detection
device = get_optimal_device()
print(f"Detected device: {device}")
assert device in ["cuda", "mps", "cpu"]
```

### Test 2: Explicit Device Selection

```python
# Test all device options
for device in ["auto", "cuda", "mps", "cpu"]:
    try:
        model = load_model_with_device("model-path", device=device)
        print(f"✅ {device} works")
    except Exception as e:
        print(f"❌ {device} failed: {e}")
```

### Test 3: MPS-Specific Behavior

```python
# On Mac
if torch.backends.mps.is_available():
    # Should return float16 for MPS
    dtype = get_optimal_dtype("mps", "bf16")
    assert dtype == torch.bfloat16 or dtype == torch.float16
```

## Best Practices Summary

1. **Always provide "auto" option** - Most users want automatic detection
2. **Include explicit options** - Power users need to override
3. **Cache models per device** - Don't reload when switching devices
4. **Log device selection** - Help users debug issues
5. **Use float16 for MPS** - Critical for performance
6. **Check hasattr for MPS** - Avoid AttributeError on non-Mac
7. **Test on all platforms** - Verify CUDA, MPS, and CPU paths

## Example: Complete Node Class

```python
import torch
import numpy as np
from typing import Dict, Any, Tuple

class AudioGenerationNode:
    """
    Complete example with full MPS/CPU support.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "Hello, this is a test."
                }),
                "device": ([
                    "auto",
                    "cuda", 
                    "mps",
                    "cpu"
                ], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Audio"
    
    def __init__(self):
        self.model = None
        self.current_config = None
    
    def generate(self, text: str, device: str, precision: str):
        # Resolve device
        if device == "auto":
            device = self._get_optimal_device()
        
        # Get optimal dtype
        dtype = self._get_dtype(device, precision)
        
        # Load model (cache by device+dtype)
        config_key = (device, str(dtype))
        if self.current_config != config_key:
            self.model = self._load_model(device, dtype)
            self.current_config = config_key
        
        # Prepare inputs
        inputs = self._tokenize(text).to(device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(inputs)
        
        # Return CPU numpy array
        audio = output.detach().cpu().numpy()
        
        return ({"waveform": audio, "sample_rate": 24000},)
    
    def _get_optimal_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _get_dtype(self, device: str, precision: str) -> torch.dtype:
        if device == "mps":
            return torch.float16 if precision == "fp32" else torch.bfloat16
        return torch.float32 if precision == "fp32" else torch.bfloat16
    
    def _load_model(self, device: str, dtype: torch.dtype):
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            "your-model",
            device_map=device,
            torch_dtype=dtype
        )
    
    def _tokenize(self, text: str) -> torch.Tensor:
        # Your tokenization logic
        pass
```

## References

- **VibeVoice-ComfyUI**: `nodes/base_vibevoice.py:34-47`
- **ComfyUI-Qwen-TTS**: `nodes.py:287-304`
- PyTorch MPS Documentation: https://pytorch.org/docs/stable/notes/mps.html
