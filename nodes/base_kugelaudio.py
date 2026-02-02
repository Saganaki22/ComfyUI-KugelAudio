"""Base node class with integrated model loading for KugelAudio.

VibeVoice-style: Each node handles loading internally with shared caching.
"""

import os
import sys
import time
import logging
import warnings
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger("KugelAudio")

# Disable torch.compile to avoid "Compiler: cl is not found" errors
# This affects audioseal watermarking which uses torch.compile internally
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
logger.debug("torch.compile disabled to avoid compiler dependency issues")

# Suppress deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.nn.utils.weight_norm.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.nn.utils.weight_norm.*")

# Suppress transformers config/ tokenizer warnings (harmless)
warnings.filterwarnings("ignore", category=UserWarning, message=".*preprocessor_config.json.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*tokenizer class.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*tokenizer class.*")

# Try to import folder_paths from ComfyUI
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    logger.warning("ComfyUI folder_paths not available")

# Try to import ComfyUI execution for progress bar
try:
    from comfy.utils import ProgressBar
    COMFYUI_UTILS_AVAILABLE = True
except ImportError:
    COMFYUI_UTILS_AVAILABLE = False

try:
    import comfy.model_management as mm
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False

# Check if SageAttention is available
# Import SageAttention patch (from ComfyUI-VibeVoice)
from .sage_attention_patch import SAGE_ATTENTION_AVAILABLE, set_sage_attention

# Supported languages
SUPPORTED_LANGUAGES = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("no", "Norwegian"),
    ("el", "Greek"),
    ("bg", "Bulgarian"),
    ("sk", "Slovak"),
    ("hr", "Croatian"),
    ("sr", "Serbian"),
    ("tr", "Turkish"),
]

# Attention implementations - matching VibeVoice format
ATTENTION_OPTIONS = [
    ("auto", "Auto (detect best available)"),
    ("sage", "SageAttention (fastest, CUDA only)"),
    ("flash_attention_2", "FlashAttention 2 (fast, CUDA only)"),
    ("sdpa", "SDPA (PyTorch optimized)"),
    ("eager", "Eager (standard/slowest)"),
]



# Shared model cache (class-level, persists across node instances)
_shared_model = None
_shared_processor = None
_shared_config = {
    "model_path": None,
    "attention_type": None,
    "use_4bit": None,
    "device": None,
}

# Model path lookup cache
_model_path_lookup: Dict[str, str] = {}


def get_comfyui_models_dir() -> str:
    """Get the ComfyUI models directory."""
    if COMFYUI_AVAILABLE:
        base_dir = folder_paths.models_dir
    else:
        base_dir = os.path.join(os.getcwd(), "models")
    return os.path.join(base_dir, "kugelaudio")


def get_available_models() -> List[Tuple[str, str]]:
    """Get list of available KugelAudio models in the models directory.
    
    Handles both direct folders and nested git clone structures.
    Returns list of (display_name, full_path) tuples.
    """
    global _model_path_lookup
    models_dir = get_comfyui_models_dir()
    available = []
    _model_path_lookup.clear()
    
    if not os.path.exists(models_dir):
        # No models found, will auto-download
        return [("kugelaudio-0-open (auto-download)", "kugelaudio/kugelaudio-0-open")]
    
    # Scan for model folders
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Check if this is a model folder directly
        if os.path.exists(os.path.join(item_path, "config.json")):
            display_name = item
            available.append((display_name, item_path))
            _model_path_lookup[display_name] = item_path
            continue
        
        # Handle nested structure from git clone (e.g., kugelaudio-0-open/kugelaudio-0-open/)
        nested_path = os.path.join(item_path, item)
        if os.path.isdir(nested_path) and os.path.exists(os.path.join(nested_path, "config.json")):
            display_name = item
            available.append((display_name, nested_path))
            _model_path_lookup[display_name] = nested_path
            logger.warning(f"Detected nested model structure at {nested_path}. Consider moving files up one level.")
            continue
    
    if not available:
        # No models found, will auto-download
        return [("kugelaudio-0-open (auto-download)", "kugelaudio/kugelaudio-0-open")]
    
    return available


def resolve_model_path(model_selection: str) -> str:
    """Resolve model selection to actual path.
    
    Args:
        model_selection: Either display name from dropdown or full path
        
    Returns:
        Full path to model directory or HF model ID
    """
    global _model_path_lookup
    
    # Check if it's in our cache (from get_available_models)
    if model_selection in _model_path_lookup:
        return _model_path_lookup[model_selection]
    
    # If it already looks like a path and exists, use it
    if os.path.exists(model_selection) and os.path.exists(os.path.join(model_selection, "config.json")):
        return model_selection
    
    # If it starts with kugelaudio/, it's a HuggingFace model ID for auto-download
    if model_selection.startswith("kugelaudio/"):
        return model_selection
    
    # Handle " (auto-download)" suffix
    clean_name = model_selection.replace(" (auto-download)", "").strip()
    if clean_name in _model_path_lookup:
        return _model_path_lookup[clean_name]
    
    # Try to find it in the models directory
    models_dir = get_comfyui_models_dir()
    potential_path = os.path.join(models_dir, model_selection)
    if os.path.exists(os.path.join(potential_path, "config.json")):
        return potential_path
    
    # Default: return as-is (may trigger auto-download if HF format)
    return model_selection


def ensure_model_downloaded(model_id: str = "kugelaudio/kugelaudio-0-open") -> str:
    """Ensure model is downloaded, returns local path.
    
    Uses HuggingFace snapshot_download (not git clone) for reliable downloads.
    Handles cases where directory exists but is empty or corrupted.
    """
    from huggingface_hub import snapshot_download
    
    models_dir = get_comfyui_models_dir()
    model_name = model_id.split("/")[-1]
    local_path = os.path.join(models_dir, model_name)
    
    # Check if already downloaded
    if os.path.exists(os.path.join(local_path, "config.json")):
        return local_path
    
    # Handle case: user manually cloned, creating nested structure
    # e.g., models/kugelaudio/kugelaudio-0-open/kugelaudio-0-open/
    nested_path = os.path.join(local_path, model_name)
    if os.path.exists(os.path.join(nested_path, "config.json")):
        logger.info(f"Found model in nested directory: {nested_path}")
        return nested_path
    
    # Download using huggingface_hub (NOT git clone)
    logger.info(f"Downloading {model_id} from HuggingFace...")
    logger.info(f"This will be saved to: {local_path}")
    os.makedirs(models_dir, exist_ok=True)
    
    # If directory exists but is empty/incomplete, snapshot_download will handle it
    try:
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,  # Resume interrupted downloads
        )
        logger.info(f"Model downloaded successfully to {downloaded_path}")
        return downloaded_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.error("If you prefer manual download, use:")
        logger.error(f"  cd {models_dir}")
        logger.error(f"  git clone https://huggingface.co/{model_id}")
        raise


class BaseKugelAudioNode:
    """Base class for KugelAudio nodes with integrated model loading.
    
    Uses shared class-level cache (VibeVoice-style) so multiple nodes
    can reuse the same loaded model.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def get_shared_model(cls):
        """Get the shared model instance."""
        global _shared_model, _shared_processor, _shared_config
        return _shared_model, _shared_processor, _shared_config
    
    @classmethod
    def set_shared_model(cls, model, processor, config):
        """Set the shared model instance."""
        global _shared_model, _shared_processor, _shared_config
        _shared_model = model
        _shared_processor = processor
        _shared_config = config.copy()
    
    @classmethod
    def clear_shared_model(cls):
        """Clear the shared model from memory."""
        global _shared_model, _shared_processor, _shared_config
        
        if _shared_model is not None:
            del _shared_model
            _shared_model = None
        
        if _shared_processor is not None:
            del _shared_processor
            _shared_processor = None
        
        _shared_config = {
            "model_path": None,
            "attention_type": None,
            "use_4bit": None,
        }
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model memory freed")
    
    def _apply_sage_attention(self, model):
        """Apply SageAttention to the loaded model.
        
        Uses the proper GPU-optimized implementation from sage_attention_patch module.
        This patches Qwen2Attention layers directly for maximum performance.
        """
        try:
            set_sage_attention(model)
            logger.info("SageAttention applied successfully (GPU-optimized kernel)")
        except Exception as e:
            logger.error(f"Failed to apply SageAttention: {e}")
            logger.warning("Continuing with standard attention implementation")
    
    def load_model(
        self,
        model_path: str,
        attention_type: str = "auto",
        use_4bit: bool = False,
        device: str = "auto",
    ) -> Tuple[Any, Any]:
        """Load model with caching (VibeVoice-style).
        
        Returns cached model if configuration matches.
        """
        from kugelaudio_open import (
            KugelAudioForConditionalGenerationInference,
            KugelAudioProcessor,
        )
        
        global _shared_model, _shared_processor, _shared_config
        
        # Check if we can use cached model
        if (_shared_model is not None and
            _shared_processor is not None and
            _shared_config.get("model_path") == model_path and
            _shared_config.get("attention_type") == attention_type and
            _shared_config.get("use_4bit") == use_4bit and
            _shared_config.get("device") == device):
            logger.debug("Using cached model")
            return _shared_model, _shared_processor
        
        # Auto-download if needed
        if model_path.startswith("kugelaudio/"):
            model_path = ensure_model_downloaded(model_path)
        elif not os.path.exists(model_path):
            raise ValueError(f"Model not found: {model_path}")
        
        # Clear existing model
        self.clear_shared_model()
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16
                logger.info("Auto-selected: CUDA device")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
                logger.info("Auto-selected: MPS device (Apple Silicon)")
                logger.warning("⚠️  MPS may cause 'mps_matmul' errors with some models.")
                logger.warning("   If you encounter crashes, manually select 'cpu' from the device dropdown.")
            else:
                device = "cpu"
                dtype = torch.float32
                logger.info("Auto-selected: CPU device")
        elif device == "cuda":
            dtype = torch.bfloat16
            logger.info(f"Using device: {device}")
        elif device == "mps":
            dtype = torch.float16
            logger.info(f"Using device: {device} (Apple Silicon)")
            logger.warning("⚠️  MPS may cause 'mps_matmul' errors with some models.")
            logger.warning("   If you encounter crashes, switch to 'cpu' from the device dropdown.")
        elif device == "cpu":
            dtype = torch.float32
            logger.info(f"Using device: {device}")
        else:
            logger.warning(f"Unknown device '{device}', falling back to auto-detection")
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.bfloat16
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float16
            else:
                device = "cpu"
                dtype = torch.float32
        
        # Prepare model kwargs
        # Use 'dtype' instead of deprecated 'torch_dtype' (transformers >= 4.56)
        try:
            from packaging import version
            import transformers
            if version.parse(transformers.__version__) >= version.parse("4.56.0"):
                model_kwargs = {"dtype": dtype}
            else:
                model_kwargs = {"torch_dtype": dtype}
        except:
            model_kwargs = {"torch_dtype": dtype}
        
        # Validate 4-bit + attention compatibility
        if use_4bit and attention_type in ["sage", "flash_attention_2"]:
            logger.warning("=" * 60)
            logger.warning("4-BIT QUANTIZATION INCOMPATIBILITY")
            logger.warning("=" * 60)
            logger.warning(f"Attention type '{attention_type}' is not compatible with 4-bit quantization.")
            logger.warning("4-bit quantization only supports: sdpa, eager")
            logger.warning("Falling back to 'sdpa' attention.")
            logger.warning("=" * 60)
            attention_type = "sdpa"
        elif use_4bit and attention_type == "auto":
            # In auto mode with 4-bit, only check for sdpa/eager
            logger.info("4-bit quantization enabled: limiting auto-selection to sdpa/eager only")
        
        # Handle attention type - following VibeVoice pattern
        use_sage_attention = False
        actual_attention = attention_type  # Track what we actually use
        
        if attention_type == "sage":
            # SageAttention requires special handling - can't be set via attn_implementation
            if device in ["cpu", "mps"]:
                logger.warning(f"SageAttention requires CUDA, not compatible with {device.upper()}, falling back to sdpa")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
            elif not SAGE_ATTENTION_AVAILABLE:
                logger.warning("SageAttention not installed, falling back to sdpa")
                logger.warning("Install with: pip install sageattention")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
            elif not torch.cuda.is_available():
                logger.warning("SageAttention requires CUDA GPU, falling back to sdpa")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
            else:
                # Don't set attn_implementation for sage, will apply after loading
                use_sage_attention = True
                actual_attention = "sage"
                logger.info("Using SageAttention (GPU-optimized)")
        elif attention_type == "auto":
            # Auto mode - check availability in priority order: sage → flash → sdpa → eager
            # But skip sage/flash if: device is CPU/MPS, 4-bit quantization is enabled, or CUDA is unavailable
            if device in ["cpu", "mps"]:
                # CPU/MPS mode: only SDPA or Eager will work (no CUDA-dependent attention)
                logger.info(f"{device.upper()} mode: Auto-selecting from sdpa/eager only (CUDA attention not supported)")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
                logger.info("Auto-selected: SDPA (CPU/MPS compatible)")
            elif use_4bit:
                # 4-bit only supports sdpa/eager
                logger.info("4-bit mode: Auto-selecting from sdpa/eager only")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
                logger.info("Auto-selected: SDPA (4-bit compatible)")
            elif SAGE_ATTENTION_AVAILABLE and torch.cuda.is_available():
                use_sage_attention = True
                actual_attention = "sage"
                logger.info("Auto-selected: SageAttention (fastest)")
            else:
                # Try flash attention
                try:
                    import flash_attn
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    actual_attention = "flash_attention_2"
                    logger.info("Auto-selected: Flash Attention 2")
                except ImportError:
                    # Fall back to sdpa
                    model_kwargs["attn_implementation"] = "sdpa"
                    actual_attention = "sdpa"
                    logger.info("Auto-selected: SDPA (PyTorch optimized)")
        elif attention_type != "auto":
            # For flash_attention_2, sdpa, eager - pass directly to transformers
            # But check CPU compatibility for flash_attention_2
            if device in ["cpu", "mps"] and attention_type == "flash_attention_2":
                logger.warning(f"Flash Attention 2 requires CUDA, not compatible with {device.upper()}, falling back to sdpa")
                model_kwargs["attn_implementation"] = "sdpa"
                actual_attention = "sdpa"
            else:
                model_kwargs["attn_implementation"] = attention_type
                actual_attention = attention_type
                logger.info(f"Using {attention_type} attention implementation")
        
        # Handle quantization - only quantize the LLM component
        # Skip: prediction_head (diffusion), speech_vae, semantic_vae for audio quality
        use_quantization = False
        
        # Disable 4-bit for non-CUDA devices (CPU/MPS don't support bitsandbytes)
        if use_4bit and device in ["cpu", "mps"]:
            logger.warning("=" * 60)
            logger.warning("4-BIT QUANTIZATION NOT AVAILABLE")
            logger.warning("=" * 60)
            logger.warning(f"4-bit quantization requires CUDA GPU.")
            logger.warning(f"Current device: '{device}' - using full precision instead.")
            logger.warning("=" * 60)
            use_4bit = False
        
        if use_4bit and device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                
                # 4-bit quantization for LLM only
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["device_map"] = "auto"
                use_quantization = True
                
                logger.info("Using 4-bit quantization (LLM only, audio components full precision)")
            except ImportError:
                logger.warning("bitsandbytes not available, using full precision")
        
        if not use_quantization:
            model_kwargs["device_map"] = device if device != "cpu" else None
        
        # Load model
        logger.info(f"Loading model...")
        start_time = time.time()
        
        model = KugelAudioForConditionalGenerationInference.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        if device == "cpu" and not use_quantization:
            model = model.to(device)
        
        model.eval()
        
        # Apply SageAttention after model loading if selected
        if use_sage_attention and SAGE_ATTENTION_AVAILABLE:
            self._apply_sage_attention(model)
        
        # Load processor
        processor = KugelAudioProcessor.from_pretrained(model_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.1f}s")
        
        # Cache the model
        config = {
            "model_path": model_path,
            "attention_type": attention_type,
            "use_4bit": use_4bit,
        }
        self.set_shared_model(model, processor, config)
        
        return model, processor
    
    def free_memory(self) -> None:
        """Free model from memory."""
        self.clear_shared_model()
    
    def _prepare_audio_from_comfyui(
        self,
        audio: Dict[str, Any],
        target_sample_rate: int = 24000,
    ) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format.
        
        Handles any sample rate and mono/stereo (converts to mono).
        """
        if audio is None:
            return None
        
        if not isinstance(audio, dict) or "waveform" not in audio:
            return None
        
        waveform = audio["waveform"]
        input_sample_rate = audio.get("sample_rate", target_sample_rate)
        
        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().float().numpy()
        else:
            audio_np = np.array(waveform, dtype=np.float32)
        
        # Handle batch dimension
        if audio_np.ndim == 3:  # (batch, channels, samples)
            audio_np = audio_np[0]
        
        # Convert to mono if stereo
        if audio_np.ndim == 2:
            if audio_np.shape[0] == 2:
                audio_np = np.mean(audio_np, axis=0)
            elif audio_np.shape[1] == 2:
                audio_np = np.mean(audio_np, axis=1)
            else:
                audio_np = audio_np.squeeze()
        
        # Resample if needed
        if input_sample_rate != target_sample_rate:
            import librosa
            audio_np = librosa.resample(
                audio_np,
                orig_sr=input_sample_rate,
                target_sr=target_sample_rate,
            )
        
        # Normalize
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val
        
        return audio_np.astype(np.float32)
    
    def _format_audio_for_comfyui(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
        output_stereo: bool = False,
    ) -> Dict[str, Any]:
        """Format audio for ComfyUI output."""
        # Ensure proper shape (batch, channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        # Convert to float32
        audio = audio.cpu().float()
        
        # Convert to stereo if requested
        if output_stereo and audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        
        return {
            "waveform": audio,
            "sample_rate": sample_rate,
        }
    
    def _check_interrupt(self):
        """Check if ComfyUI has requested interruption."""
        if INTERRUPTION_SUPPORT:
            try:
                mm.throw_exception_if_processing_interrupted()
            except Exception:
                raise
