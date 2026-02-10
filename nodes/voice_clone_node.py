"""Voice cloning node for KugelAudio."""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from .base_kugelaudio import (
    BaseKugelAudioNode,
    get_available_models,
    resolve_model_path,
    ATTENTION_OPTIONS,
    SUPPORTED_LANGUAGES,
    INTERRUPTION_SUPPORT,
)
from .text_utils import split_text_into_chunks

# ComfyUI ProgressBar
try:
    from comfy.utils import ProgressBar
    COMFYUI_PROGRESS_AVAILABLE = True
except ImportError:
    COMFYUI_PROGRESS_AVAILABLE = False

logger = logging.getLogger("KugelAudio")


class KugelAudioVoiceCloneNode(BaseKugelAudioNode):
    """Clone a voice using reference audio."""
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def INPUT_TYPES(cls):
        available_models = get_available_models()
        model_choices = [name for name, _ in available_models]
        default_model = model_choices[0] if model_choices else "kugelaudio-0-open"
        
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! I am speaking with a cloned voice.",
                    "tooltip": "Text to convert to speech using the cloned voice.",
                }),
                "model": (model_choices, {
                    "default": default_model,
                    "tooltip": "Select model from ComfyUI/models/kugelaudio/ (auto-downloads on first run)",
                }),
                "attention_type": ([opt[0] for opt in ATTENTION_OPTIONS], {
                    "default": "auto",
                    "tooltip": "Attention implementation. Auto detects best available. SageAttention/FlashAttention require CUDA.",
                }),
                "use_4bit": ("BOOLEAN", {
                    "default": False,
                    "label_on": "4-bit (BNB)",
                    "label_off": "Full Precision",
                    "tooltip": "Quantize the LLM to 4-bit using bitsandbytes. Reduces VRAM from ~19GB to ~8GB. Audio components stay at full precision. Requires CUDA GPU - automatically disabled for CPU/MPS devices.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. Higher = more adherence to text. Default: 3.0",
                }),
                "max_new_tokens": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 256,
                    "tooltip": "Maximum tokens to generate. Increase for longer text.",
                }),
                "language": ([opt[0] for opt in SUPPORTED_LANGUAGES], {
                    "default": "auto",
                    "tooltip": "Optional language hint. Model auto-detects if set to Auto.",
                }),
                "keep_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in VRAM after generation. Disable to free memory (slower subsequent runs).",
                }),
                "output_stereo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output stereo audio (duplicates mono channel). Use if your workflow expects stereo.",
                }),
                "device": (["auto", "cuda", "mps", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to use for inference. Auto detects best available. Select 'cpu' for Apple Silicon MPS compatibility if you get errors.",
                }),
            },
            "optional": {
                "voice_prompt": ("AUDIO", {
                    "tooltip": "Reference audio to clone (5-30 seconds recommended for best quality). Any sample rate, mono/stereo accepted (auto-converted).",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "Random seed for reproducible generation.",
                }),
                "max_words_per_chunk": ("INT", {
                    "default": 250,
                    "min": 100,
                    "max": 500,
                    "step": 50,
                    "tooltip": "Split long text into chunks at sentence boundaries (100-500, default 250). Helps with very long text.",
                }),
                "do_sample": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable sampling for more varied output. Disabled = deterministic.",
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Sampling temperature (only used if do_sample=True). KugelAudio does not support top_p/top_k.",
                }),
                "disable_watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Disable audio watermarking. Enable this if you experience stuttering/micro-freezes in the generated audio.",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone_voice"
    CATEGORY = "KugelAudio"
    DESCRIPTION = "Clone a voice using reference audio with KugelAudio"
    
    def clone_voice(
        self,
        text: str,
        model: str,
        attention_type: str,
        use_4bit: bool,
        cfg_scale: float,
        max_new_tokens: int,
        language: str,
        keep_loaded: bool,
        output_stereo: bool,
        device: str,
        voice_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        max_words_per_chunk: int = 250,
        do_sample: bool = False,
        temperature: float = 1.0,
        disable_watermark: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        """Clone voice using reference audio."""
        try:
            # Check for interruption
            self._check_interrupt()
            
            # Validate text
            if not text or not text.strip():
                raise ValueError("No text provided")
            
            # Set seed for reproducibility if provided
            if seed >= 0:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                logger.info(f"Using seed: {seed}")
            
            # Progress bar setup (Qwen3-TTS style - stage based)
            # Calculate total stages: 1 (load) + num_chunks (generation) + 1 (finalize)
            if max_words_per_chunk > 0:
                chunks = split_text_into_chunks(text, max_words_per_chunk)
            else:
                chunks = [text]
            
            total_stages = 1 + len(chunks) + 1  # Load + Generate chunks + Finalize
            
            if COMFYUI_PROGRESS_AVAILABLE:
                pbar = ProgressBar(total_stages)
                logger.info(f"Progress: 0/{total_stages} - Loading model...")
            else:
                pbar = None
            
            # Load model
            model_obj, processor = self.load_model(
                model_path=resolve_model_path(model),
                attention_type=attention_type,
                use_4bit=use_4bit,
                device=device,
            )
            
            if pbar:
                pbar.update_absolute(1, total_stages, None)
            
            # Get device
            device = next(model_obj.parameters()).device
            
            # Prepare voice prompt if provided
            voice_audio = None
            if voice_prompt is not None:
                voice_audio = self._prepare_audio_from_comfyui(voice_prompt)
                if voice_audio is not None:
                    logger.info(f"Using voice prompt: {len(voice_audio)/24000:.1f}s")
            
            # Generate audio for each chunk
            audio_tensors = []
            num_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                if num_chunks > 1:
                    logger.info(f"Processing chunk {i+1}/{num_chunks}...")

                # Check for interruption
                self._check_interrupt()

                # Prepare inputs
                if voice_audio is not None:
                    inputs = processor(
                        text=chunk,
                        voice_prompt=voice_audio,
                        return_tensors="pt"
                    )
                else:
                    inputs = processor(text=chunk, return_tensors="pt")

                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                # Check for interruption before generation
                self._check_interrupt()

                # Estimate tokens for user information
                text_word_count = len(chunk.split())
                estimated_tokens = int(text_word_count * 2.5)
                logger.info(f"Generating audio (~{estimated_tokens} estimated tokens, max {max_new_tokens} allowed)...")

                # Generate (KugelAudio uses internal tqdm, we use stage-based progress)
                # Disable watermark for individual chunks to avoid boundary artifacts (unless disabled by user)
                with torch.no_grad():
                    outputs = model_obj.generate(
                        **inputs,
                        cfg_scale=cfg_scale,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        show_progress=True,  # Enable tqdm for CLI progress
                        apply_watermark=(num_chunks == 1 and not disable_watermark),  # Only watermark if single chunk and not disabled
                    )

                # Update progress bar after each chunk
                if pbar:
                    pbar.update_absolute(1 + i + 1, total_stages, None)

                # Check for interruption after generation
                self._check_interrupt()

                # Log completion
                logger.info(f"Audio generated successfully")

                # Extract audio - watermark applied only if single chunk
                chunk_audio = outputs.speech_outputs[0]
                audio_tensors.append(chunk_audio)

            # Finalize progress
            if pbar:
                pbar.update_absolute(total_stages, total_stages, None)

            # Concatenate all chunks
            if len(audio_tensors) > 1:
                full_audio = torch.cat(audio_tensors, dim=-1)
                # Apply watermark once to full audio to avoid chunk boundary artifacts (unless disabled)
                if not disable_watermark:
                    logger.info("Applying watermark to full audio...")
                    full_audio = model_obj._apply_watermark(full_audio, sample_rate=24000)
                else:
                    logger.info("Watermarking disabled by user")
            else:
                full_audio = audio_tensors[0]
            
            # Format for ComfyUI
            audio_dict = self._format_audio_for_comfyui(
                full_audio,
                sample_rate=24000,
                output_stereo=output_stereo,
            )
            
            # Free memory if not keeping loaded
            if not keep_loaded:
                self.free_memory()
            
            return (audio_dict,)
            
        except Exception as e:
            # Check if this is an interruption
            if INTERRUPTION_SUPPORT:
                from comfy.model_management import InterruptProcessingException
                if isinstance(e, InterruptProcessingException):
                    raise
            
            logger.error(f"Voice cloning failed: {e}")
            raise Exception(f"Voice cloning failed: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache key for ComfyUI."""
        return hash(str(kwargs))
