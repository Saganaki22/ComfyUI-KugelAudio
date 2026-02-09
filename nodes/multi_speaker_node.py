"""Multi-speaker node for KugelAudio."""

import re
import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional

from .base_kugelaudio import (
    BaseKugelAudioNode,
    get_available_models,
    resolve_model_path,
    ATTENTION_OPTIONS,
    SUPPORTED_LANGUAGES,
    INTERRUPTION_SUPPORT,
)

# ComfyUI ProgressBar
try:
    from comfy.utils import ProgressBar
    COMFYUI_PROGRESS_AVAILABLE = True
except ImportError:
    COMFYUI_PROGRESS_AVAILABLE = False

logger = logging.getLogger("KugelAudio")


class KugelAudioMultiSpeakerNode(BaseKugelAudioNode):
    """Generate multi-speaker conversations.
    
    Uses native KugelAudio format:
    Speaker 0: Hello, I'm speaker one.
    Speaker 1: Hi there, I'm speaker two.
    
    Each speaker line is processed separately with its voice sample if provided.
    """
    
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
                    "default": "Speaker 1: Hello, I'm the first speaker.\nSpeaker 2: Hi there, I'm the second speaker.\nSpeaker 3: I'm the third speaker.\nSpeaker 4: And I'm the fourth.\nSpeaker 5: Greetings from speaker five.\nSpeaker 6: Hello from speaker six!",
                    "tooltip": "Use 'Speaker N: text' format (N=1-6). Internally converted to 0-5 for KugelAudio.",
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
                    "tooltip": "Maximum tokens per speaker line. Increase for longer lines.",
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
                "speaker1_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 1 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "speaker2_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 2 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "speaker3_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 3 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "speaker4_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 4 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "speaker5_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 5 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "speaker6_voice": ("AUDIO", {
                    "tooltip": "Voice sample for Speaker 6 (optional). Any sample rate, mono/stereo accepted.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "Random seed for reproducible generation.",
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
                "pause_between_speakers": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Add pause (in seconds) between each speaker for natural pacing.",
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_multi_speaker"
    CATEGORY = "KugelAudio"
    DESCRIPTION = "Generate multi-speaker conversations (up to 6 speakers)"
    
    def generate_multi_speaker(
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
        speaker1_voice: Optional[Dict[str, Any]] = None,
        speaker2_voice: Optional[Dict[str, Any]] = None,
        speaker3_voice: Optional[Dict[str, Any]] = None,
        speaker4_voice: Optional[Dict[str, Any]] = None,
        speaker5_voice: Optional[Dict[str, Any]] = None,
        speaker6_voice: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        do_sample: bool = False,
        temperature: float = 1.0,
        pause_between_speakers: float = 0.2,
    ) -> Tuple[Dict[str, Any]]:
        """Generate multi-speaker audio - processes each line separately with voice cloning."""
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
            
            # Parse speakers from text (accepts Speaker 1-6, converts to 0-5 internally)
            speaker_pattern = r'^Speaker\s+(\d+)\s*:\s*(.*)$'
            lines = text.strip().split('\n')
            parsed_lines = []
            speaker_ids = set()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(speaker_pattern, line, re.IGNORECASE)
                if match:
                    speaker_id = int(match.group(1))
                    text_content = match.group(2).strip()
                    
                    # Convert 1-6 to 0-5 for KugelAudio
                    if 1 <= speaker_id <= 6:
                        internal_speaker_id = speaker_id - 1
                    else:
                        raise ValueError(f"Speaker {speaker_id} is invalid. Use Speaker 1-6.")
                    
                    parsed_lines.append((internal_speaker_id, text_content))
                    speaker_ids.add(internal_speaker_id)
            
            if not parsed_lines:
                raise ValueError("No valid speaker lines found. Use format: 'Speaker N: text' (N=1-6)")
            
            # Check speaker limit
            if max(speaker_ids) > 5:
                raise ValueError("Maximum 6 speakers supported (Speaker 1-6)")
            
            logger.info(f"Found {len(parsed_lines)} lines from {len(speaker_ids)} speakers (Speaker 1-6 → 0-5)")
            
            # Progress bar setup
            # Each line is processed separately: 1 (load) + num_lines (generation) + 1 (finalize)
            total_stages = 1 + len(parsed_lines) + 1
            
            if COMFYUI_PROGRESS_AVAILABLE:
                pbar = ProgressBar(total_stages)
                logger.info(f"Progress: 0/{total_stages} - Loading model...")
            else:
                pbar = None
            
            # Resolve model path from display name
            model_path = resolve_model_path(model)
            
            # Load model
            model_obj, processor = self.load_model(
                model_path=model_path,
                attention_type=attention_type,
                use_4bit=use_4bit,
                device=device,
            )
            
            if pbar:
                pbar.update_absolute(1, total_stages, None)
            
            # Get device
            device = next(model_obj.parameters()).device
            
            # Prepare voice samples dictionary (convert Speaker 1-6 to 0-5)
            voice_samples = {
                0: self._prepare_audio_from_comfyui(speaker1_voice) if speaker1_voice else None,
                1: self._prepare_audio_from_comfyui(speaker2_voice) if speaker2_voice else None,
                2: self._prepare_audio_from_comfyui(speaker3_voice) if speaker3_voice else None,
                3: self._prepare_audio_from_comfyui(speaker4_voice) if speaker4_voice else None,
                4: self._prepare_audio_from_comfyui(speaker5_voice) if speaker5_voice else None,
                5: self._prepare_audio_from_comfyui(speaker6_voice) if speaker6_voice else None,
            }
            
            # Generate audio for each line separately
            audio_segments = []
            for i, (speaker_id, text_content) in enumerate(parsed_lines):
                logger.info(f"Processing line {i+1}/{len(parsed_lines)}: Speaker {speaker_id}")
                
                # Check for interruption
                self._check_interrupt()
                
                # Prepare inputs with voice prompt if available
                voice_audio = voice_samples.get(speaker_id)
                if voice_audio is not None:
                    display_speaker = speaker_id + 1  # Convert back to 1-6 for display
                    logger.info(f"  Using voice sample for Speaker {display_speaker}")
                    inputs = processor(
                        text=text_content,
                        voice_prompt=voice_audio,
                        return_tensors="pt"
                    )
                else:
                    display_speaker = speaker_id + 1
                    logger.info(f"  Using base TTS for Speaker {display_speaker} (no voice sample)")
                    inputs = processor(
                        text=text_content,
                        return_tensors="pt"
                    )
                
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                
                # Check for interruption before generation
                self._check_interrupt()
                
                # Estimate tokens
                text_word_count = len(text_content.split())
                estimated_tokens = int(text_word_count * 2.5)
                logger.info(f"  Generating (~{estimated_tokens} estimated tokens)...")
                
                # Generate
                # Disable watermark for individual segments to avoid boundary artifacts
                with torch.no_grad():
                    outputs = model_obj.generate(
                        **inputs,
                        cfg_scale=cfg_scale,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        show_progress=True,
                        apply_watermark=False,  # Will apply once after concatenation
                    )
                
                # Update progress after each line
                if pbar:
                    pbar.update_absolute(1 + i + 1, total_stages, None)
                
                # Check for interruption after generation
                self._check_interrupt()
                
                # Extract audio - watermark already applied by model
                segment_audio = outputs.speech_outputs[0]
                audio_segments.append(segment_audio)
                logger.info(f"  Line {i+1} generated successfully")
            
            # Concatenate all audio segments with pause between speakers
            sample_rate = 24000
            if pause_between_speakers > 0 and len(audio_segments) > 1:
                silence_samples = int(pause_between_speakers * sample_rate)
                
                # Create silence with same dimensions as audio segments (1D)
                silence = torch.zeros(silence_samples, dtype=audio_segments[0].dtype)
                
                segments_with_pause = []
                for i, audio in enumerate(audio_segments):
                    # Ensure audio is 1D
                    if audio.dim() > 1:
                        audio = audio.squeeze()
                    segments_with_pause.append(audio)
                    if i < len(audio_segments) - 1:
                        segments_with_pause.append(silence)
                
                full_audio = torch.cat(segments_with_pause, dim=-1)
                logger.info(f"Concatenated {len(audio_segments)} segments with {pause_between_speakers}s pause between speakers")
            else:
                if len(audio_segments) > 1:
                    full_audio = torch.cat(audio_segments, dim=-1)
                    logger.info(f"Concatenated {len(audio_segments)} audio segments")
                else:
                    full_audio = audio_segments[0]
            
            # Apply watermark once to full audio to avoid segment boundary artifacts
            logger.info("Applying watermark to full audio...")
            full_audio = model_obj._apply_watermark(full_audio, sample_rate=24000)
            
            # Format for ComfyUI
            audio_dict = self._format_audio_for_comfyui(
                full_audio,
                sample_rate=24000,
                output_stereo=output_stereo,
            )
            
            # Finalize progress
            if pbar:
                pbar.update_absolute(total_stages, total_stages, None)
            
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
            
            logger.error(f"Multi-speaker generation failed: {e}")
            raise Exception(f"Multi-speaker generation failed: {str(e)}")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Cache key for ComfyUI."""
        return hash(str(kwargs))
