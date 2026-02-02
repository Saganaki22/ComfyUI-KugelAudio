"""KugelAudio - Open Source Text-to-Speech Model

KugelAudio is a state-of-the-art neural text-to-speech model that generates
natural, expressive speech from text with voice cloning capabilities.

Example:
    >>> from kugelaudio import KugelAudioForConditionalGenerationInference
    >>> from transformers import AutoModel
    >>>
    >>> # Load the model
    >>> model = AutoModel.from_pretrained("kugelaudio/kugelaudio-0-open")
"""

__version__ = "0.1.0"

from .configs import (
    KugelAudioAcousticTokenizerConfig,
    KugelAudioConfig,
    KugelAudioDiffusionHeadConfig,
    KugelAudioSemanticTokenizerConfig,
)
from .models import (
    KugelAudioAcousticTokenizerModel,
    KugelAudioDiffusionHead,
    KugelAudioForConditionalGeneration,
    KugelAudioForConditionalGenerationInference,
    KugelAudioModel,
    KugelAudioPreTrainedModel,
    KugelAudioSemanticTokenizerModel,
)
from .processors import KugelAudioProcessor
from .schedule import DPMSolverMultistepScheduler
from .watermark import AudioWatermark


# Lazy imports for optional components
def launch_ui(*args, **kwargs):
    """Launch the Gradio web interface."""
    try:
        from .ui import launch_ui as _launch_ui

        return _launch_ui(*args, **kwargs)
    except ImportError:
        raise ImportError(
            "Gradio is required for the web interface. " "Install it with: pip install gradio"
        )


__all__ = [
    # Version
    "__version__",
    # Configs
    "KugelAudioConfig",
    "KugelAudioAcousticTokenizerConfig",
    "KugelAudioSemanticTokenizerConfig",
    "KugelAudioDiffusionHeadConfig",
    # Models
    "KugelAudioModel",
    "KugelAudioPreTrainedModel",
    "KugelAudioForConditionalGeneration",
    "KugelAudioForConditionalGenerationInference",
    "KugelAudioAcousticTokenizerModel",
    "KugelAudioSemanticTokenizerModel",
    "KugelAudioDiffusionHead",
    # Scheduler
    "DPMSolverMultistepScheduler",
    # Processors
    "KugelAudioProcessor",
    # Watermark
    "AudioWatermark",
    # UI
    "launch_ui",
]
