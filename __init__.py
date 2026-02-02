"""ComfyUI nodes for KugelAudio - Open-source text-to-speech with voice cloning.

This package provides ComfyUI integration for KugelAudio TTS model.
"""

__version__ = "1.0.0"
__author__ = "KugelAudio"

import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any

# Setup minimal logging
logger = logging.getLogger("KugelAudio")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[KugelAudio] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def install_local_kugelaudio():
    """Auto-install kugelaudio-open from the local bundled folder."""
    try:
        import kugelaudio_open
        return True  # Already installed
    except ImportError:
        pass
    
    # Find the local kugelaudio-open folder
    current_dir = Path(__file__).parent
    local_kugelaudio = current_dir / "kugelaudio-open"
    
    if not local_kugelaudio.exists():
        logger.error("kugelaudio-open folder not found in custom node")
        return False
    
    if not (local_kugelaudio / "setup.py").exists() and not (local_kugelaudio / "pyproject.toml").exists():
        logger.error("kugelaudio-open folder exists but missing setup files")
        return False
    
    # Install using the current Python interpreter
    logger.info("Installing kugelaudio-open from bundled folder...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", str(local_kugelaudio)],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info("kugelaudio-open installed successfully!")
            return True
        else:
            logger.error(f"Failed to install kugelaudio-open: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error installing kugelaudio-open: {e}")
        return False


def ensure_dependencies():
    """Ensure required dependencies are installed."""
    # First, try to auto-install kugelaudio-open if missing
    if not install_local_kugelaudio():
        pass  # Will check other deps and report
    
    missing_packages = []
    
    try:
        import torch
    except ImportError:
        missing_packages.append("torch")
    
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.40.0"):
            logger.warning(f"Transformers {transformers.__version__} may be too old. Recommended: >=4.40.0")
    except ImportError:
        missing_packages.append("transformers")
    
    try:
        import kugelaudio_open
    except ImportError:
        missing_packages.append("kugelaudio-open (auto-install failed)")
    
    if missing_packages:
        logger.error("=" * 60)
        logger.error("KUGELAUDIO MISSING DEPENDENCIES")
        logger.error("=" * 60)
        for pkg in missing_packages:
            logger.error(f"  - {pkg}")
        logger.error("")
        logger.error("To install missing packages manually:")
        logger.error("  pip install torch transformers")
        logger.error("")
        logger.error("For embedded Python (ComfyUI portable):")
        logger.error("  python_embeded\\python.exe -m pip install torch transformers")
        logger.error("=" * 60)
        return False
    
    return True


# Node mappings
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# Register nodes if dependencies are available
if ensure_dependencies():
    try:
        from .nodes.tts_node import KugelAudioTTSNode
        from .nodes.voice_clone_node import KugelAudioVoiceCloneNode
        from .nodes.multi_speaker_node import KugelAudioMultiSpeakerNode
        from .nodes.watermark_node import KugelAudioWatermarkNode
        NODE_CLASS_MAPPINGS["KugelAudioTTSNode"] = KugelAudioTTSNode
        NODE_DISPLAY_NAME_MAPPINGS["KugelAudioTTSNode"] = "KugelAudio TTS"
        
        NODE_CLASS_MAPPINGS["KugelAudioVoiceCloneNode"] = KugelAudioVoiceCloneNode
        NODE_DISPLAY_NAME_MAPPINGS["KugelAudioVoiceCloneNode"] = "KugelAudio Voice Clone"
        
        NODE_CLASS_MAPPINGS["KugelAudioMultiSpeakerNode"] = KugelAudioMultiSpeakerNode
        NODE_DISPLAY_NAME_MAPPINGS["KugelAudioMultiSpeakerNode"] = "KugelAudio Multi-Speaker"
        
        NODE_CLASS_MAPPINGS["KugelAudioWatermarkNode"] = KugelAudioWatermarkNode
        NODE_DISPLAY_NAME_MAPPINGS["KugelAudioWatermarkNode"] = "KugelAudio Watermark Check"
        
        logger.info(f"KugelAudio nodes registered successfully (v{__version__})")
        
    except Exception as e:
        logger.error(f"Failed to register KugelAudio nodes: {e}")
else:
    logger.warning("KugelAudio nodes unavailable - missing dependencies")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']
