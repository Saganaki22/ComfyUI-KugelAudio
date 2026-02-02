<h1 align="center">ComfyUI-KugelAudio</h1>

<p align="center">
  <strong>ComfyUI nodes for KugelAudio</strong> - Open-source text-to-speech with voice cloning for 24 European languages<br>
  Powered by an AR + Diffusion architecture
</p>

<p align="center">
  <a href="https://github.com/Kugelaudio/kugelaudio-open">
    <img src="https://img.shields.io/badge/GitHub-Repository-black" alt="GitHub Repository">
  </a>
  <a href="https://huggingface.co/kugelaudio/kugelaudio-0-open">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face_Model-blue" alt="HuggingFace Model">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
  <a href="https://github.com/comfyanonymous/ComfyUI">
    <img src="https://img.shields.io/badge/ComfyUI-Supported-orange" alt="ComfyUI">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</p>

<img width="2271" height="972" alt="Screenshot 2026-02-02 050108" src="https://github.com/user-attachments/assets/01c0c3c9-36c3-423c-a69f-03b783b7342b" />

## Features

- **Single Speaker TTS**: Convert text to speech
- **Voice Cloning**: Clone any voice from reference audio (5-30 seconds)
- **Multi-Speaker**: Generate conversations with up to 6 speakers (Speaker 1-6)
- **Natural Pacing**: Configurable pause (0.0-2.0s) between speakers
- **Watermark Detection**: All output contains inaudible watermark (AudioSeal)
- **24 European Languages**: English, German, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Czech, Romanian, Hungarian, Swedish, Danish, Finnish, Norwegian, Greek, Bulgarian, Slovak, Croatian, Serbian, Turkish
- **4-bit Quantization**: Reduce VRAM from ~19GB to **~8GB (4-bit)**
- **Multiple Attention Types**: Auto/SageAttention/FlashAttention/SDPA/Eager
- **Progress Tracking**: Real-time progress bars for long generations
- **Text Chunking**: Automatic sentence-boundary splitting for long texts

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Click "Install Custom Nodes"
3. Search for "KugelAudio"
4. Click Install

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kugelaudio/ComfyUI-KugelAudio.git
```

**Bundled package:** The `kugelaudio-open` folder is included and must be installed. ComfyUI will try to auto-install it on first launch. If that fails, install manually (see below).

### Platform Support

✅ **Windows** (Standard & Portable)  
✅ **macOS** (Intel & Apple Silicon)  
✅ **Linux** (Standard & Portable)  

The auto-installer detects your Python environment automatically and installs to the correct location.

### Manual Installation (if auto-install fails)

If you see errors on startup about missing `kugelaudio-open` package, install manually:

**Windows Portable (recommended):**
```bash
cd ComfyUI/custom_nodes/ComfyUI-KugelAudio
..\..\..\python_embeded\python.exe -m pip install ./kugelaudio-open
```
Or double-click `install_portable.bat` in the ComfyUI-KugelAudio folder.

> **Having issues?** See [Troubleshooting](#troubleshooting) for more solutions.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (usually included with ComfyUI)
- Transformers 4.40+ (usually included with ComfyUI)
- **~19GB VRAM** for full precision (7B model)
- **~8GB VRAM with 4-bit quantization** (requires bitsandbytes)
- CUDA-capable GPU recommended (CPU/MPS supported but slower)

<details>
<summary><b>VRAM Comparison</b></summary>

| Mode | VRAM | Quality | Attention Types |
|------|------|---------|-----------------|
| Full Precision | ~19GB | Best | All (Sage/Flash/SDPA/Eager) |
| 4-bit Quantization | **~8GB** | Slight reduction | SDPA/Eager only |

</details>

<details>
<summary><b>For 4-bit Quantization (Lower VRAM)</b></summary>

4-bit quantization requires `bitsandbytes`. If you encounter issues:

```bash
# Standard installation
pip install bitsandbytes

# Or if pip install fails, try:
pip install --index-url https://pypi.org/simple/ bitsandbytes
```

**Note:** 4-bit quantization only supports SDPA and Eager attention types.

</details>

<details>
<summary><b>Node Reference</b></summary>

### KugelAudio TTS

Generate speech from text with full control over generation parameters.

**Inputs:**
- `text`: Text to synthesize
- `model`: Model selection (auto-downloads on first run)
- `attention_type`: Attention implementation (auto/sage_attn/flash_attn/sdpa/eager)
- `use_4bit`: Enable 4-bit quantization (~8GB VRAM, SDPA/Eager only)
- `cfg_scale`: Guidance scale (1.0-10.0, default 3.0) - higher = more adherence to text
- `max_new_tokens`: Max generation length (512-4096, default 2048)
- `language`: Optional language hint (auto-detects if not set)
- `keep_loaded`: Keep model in VRAM (faster subsequent runs)
- `output_stereo`: Output stereo audio
- `seed`: Random seed for reproducibility (default 42)
- `max_words_per_chunk`: Split long text at sentence boundaries (100-500, default 250)
- `do_sample`: Enable sampling for varied output (default False)
- `temperature`: Sampling temperature (0.1-2.0, default 1.0)

### KugelAudio Voice Clone

Clone any voice using a short reference audio sample (5-30 seconds recommended).

**Same inputs as TTS plus:**
- `voice_prompt`: Reference audio file for voice cloning
- Higher quality reference = better voice similarity

### KugelAudio Multi-Speaker

Generate conversations with up to 6 speakers with automatic pause between speakers.

**Inputs:**
- `text`: Conversation text (use `Speaker N:` format, N=1-6)
- `pause_between_speakers`: Silence between speaker turns (0.0-2.0 seconds, default 0.2s)
- Voice inputs for each speaker (optional)
- All TTS options (cfg_scale, attention type, etc.)

**Text Format:**
```
Speaker 1: Hello, I'm the first speaker.
Speaker 2: Hi there, I'm the second speaker.
Speaker 3: I'm the third speaker!
Speaker 4: And I'm the fourth.
Speaker 5: Adding a fifth voice here!
Speaker 6: And the sixth speaker!
```

**Optional voice inputs:**
- `speaker1_voice` through `speaker6_voice`: Voice samples for each speaker

### KugelAudio Watermark Check

All KugelAudio output contains an inaudible watermark using Facebook's AudioSeal technology. This node detects whether audio was generated by KugelAudio.

**Returns:**
- `detected`: String ("Detected" / "Not Detected")
- `confidence`: Float (0.0-1.0)

**Audio Format:**
- **Input:** Any sample rate, mono or stereo (auto-converted)
- **Output:** 24kHz mono (optionally stereo)

</details>

## Quantization Details

<details>
<summary><b>Click to expand</b></summary>

The 4-bit toggle quantizes the **LLM component** (7B parameters), keeping the diffusion head and tokenizers at full precision for best audio quality.

**Attention Type Compatibility:**

| Mode | Available Attention Types |
|------|---------------------------|
| Full Precision | Auto → SageAttention → FlashAttention 2 → SDPA → Eager |
| 4-bit | Auto (falls back to SDPA) → SDPA → Eager only |

**Tips:**
- **SageAttention**: Fastest (CUDA only, GPU-optimized kernels)
- **FlashAttention 2**: Fast (CUDA only)
- **SDPA**: PyTorch optimized (all platforms)
- **Eager**: Standard/slowest (all platforms, required for 4-bit)

</details>

## Model Auto-Download

On first run:
1. The **kugelaudio-open package** auto-installs from the bundled folder
2. The **model** (kugelaudio-0-open, 7B parameters) automatically downloads to `ComfyUI/models/kugelaudio/`

Both happen automatically on first generation.

<details>
<summary><b>Benchmark Results</b></summary>

KugelAudio achieves state-of-the-art performance, beating industry leaders including ElevenLabs in rigorous human preference testing.

**Human Preference Benchmark (A/B Testing):** 339 human evaluations comparing KugelAudio against leading TTS models.

**OpenSkill Ranking:**

| Rank | Model | Score | Record | Win Rate |
|------|-------|-------|--------|----------|
| 🥇 1 | **KugelAudio** | **26** | 71W / 20L / 23T | **78.0%** |
| 🥈 2 | ElevenLabs Multi v2 | 25 | 56W / 34L / 22T | 62.2% |
| 🥉 3 | ElevenLabs v3 | 21 | 64W / 34L / 16T | 65.3% |
| 4 | Cartesia | 21 | 55W / 38L / 19T | 59.1% |
| 5 | VibeVoice | 10 | 30W / 74L / 8T | 28.8% |
| 6 | CosyVoice v3 | 9 | 15W / 91L / 8T | 14.2% |

**Model Specs:**

| Model | Parameters | Quality | RTF | VRAM |
|-------|------------|---------|-----|------|
| kugelaudio-0-open | 7B | Best | 1.00 | ~19GB / ~8GB (4-bit) |

*RTF = Real-Time Factor (generation time / audio duration).*

</details>

<details>
<summary><b>Troubleshooting</b></summary>

### Voice cloning failed: 'Qwen2Config' object has no attribute 'pad_token_id'
## run install_portable.bat in ComfyUI\custom_nodes\ComfyUI-KugelAudio

### Out of Memory (OOM) Errors

1. **Enable 4-bit quantization**: Reduces VRAM from ~19GB to ~8GB
2. **Use SDPA or Eager attention**: Required with 4-bit mode
3. **Reduce max_words_per_chunk**: Lower chunk size reduces peak memory
4. **Restart ComfyUI**: Sometimes VRAM doesn't release properly
5. **Close other GPU applications**: Free up GPU memory
6. **Use SageAttention** (CUDA only): Most memory-efficient attention type

### bitsandbytes Installation Fails

**Windows:**
```bash
pip install bitsandbytes
```

**Linux:**
```bash
pip install bitsandbytes
```

**macOS (Apple Silicon):**
```bash
pip install bitsandbytes
```

Note: 4-bit quantization on macOS with MPS may have limited compatibility.

### Model Download Fails

1. Check internet connection
2. Try manual download:
   ```bash
   huggingface-cli download kugelaudio/kugelaudio-0-open --local-dir ComfyUI/models/kugelaudio/kugelaudio-0-open
   ```
3. Set HF_TOKEN environment variable if using gated model

### Manual Package Installation (Portable ComfyUI)

If the auto-installer fails or you need to reinstall the bundled package:

**Find your Python path:**
- Standard: `python`
- Portable Windows: `python_embeded\python.exe`

**Install the bundled package:**
```bash
# Navigate to the custom node folder
cd ComfyUI/custom_nodes/ComfyUI-KugelAudio

# Install using portable Python (replace /path/to/ComfyUI with your actual path) or run .bat file
C:\path\to\ComfyUI\python_embeded\python.exe -m pip install ./kugelaudio-open

# For standard Python installation
python -m pip install ./kugelaudio-open

```

**Verify installation:**
```bash
C:\path\to\ComfyUI\python_embeded\python.exe -c "import kugelaudio_open; print('kugelaudio-open installed successfully')"
```

**Note:** The bundled package is located at `ComfyUI/custom_nodes/ComfyUI-KugelAudio/kugelaudio-open/`

### Audio Quality Issues

1. **Static/noise**: Disable 4-bit quantization, use full precision
2. **Robot voice**: Increase cfg_scale (try 3.0-5.0)
3. **Clipping/distortion**: Lower cfg_scale (try 2.0-3.0)
4. **Slow generation**: Use SageAttention with CUDA GPU

### Attention Type Warnings

If you see warnings about attention types:
- 4-bit mode requires SDPA or Eager
- Auto mode will automatically select the best compatible option
- SageAttention and FlashAttention require CUDA

### CPU Mode (No GPU)

For systems without CUDA:
- Set attention_type to "eager"
- 4-bit quantization still works with SDPA
- Expect significantly slower generation

### Long Text Not Processing

1. Set `max_words_per_chunk` to split long text (recommended: 200-300)
2. Check for proper sentence-ending punctuation
3. Progress bar shows stage completion in ComfyUI console

### Multi-Speaker Not Working

1. Format text exactly: `Speaker N: Text` (N = 1-6)
2. Ensure voice inputs are provided if using voice cloning
3. Each line must have a speaker prefix
4. Maximum 6 speakers (1-6)

### Watermark Detection Issues

- Watermark detection works on generated audio only
- Very short audio clips may have reduced detection accuracy
- Output shows "Detected" or "Not Detected" as string

</details>

## Intended Use

<details>
<summary><b>Click to expand</b></summary>

### ✅ Appropriate Uses

- **Accessibility**: Text-to-speech for visually impaired users
- **Content Creation**: Podcasts, videos, audiobooks, e-learning
- **Voice Assistants**: Chatbots and virtual assistants
- **Language Learning**: Pronunciation practice and language education
- **Creative Projects**: With proper consent and attribution

### ❌ Prohibited Uses

- Creating deepfakes or misleading content
- Impersonating individuals without explicit consent
- Fraud, deception, or scams
- Harassment or abuse
- Any illegal activities

### Limitations

- **VRAM Requirements**: Requires ~19GB VRAM for full precision, ~8GB with **4-bit quantization**
- **Speed**: Approximately 1.0x real-time on modern GPUs
- **Voice Cloning Quality**: Best results with 5-30 seconds of clear reference audio
- **Language Quality Variation**: Quality may vary across languages based on training data distribution

</details>

## License

MIT License - Same as [KugelAudio](https://github.com/Kugelaudio/kugelaudio-open)

## Acknowledgments

This model would not have been possible without the contributions of many individuals and organizations:

- [Microsoft VibeVoice Team](https://github.com/microsoft/VibeVoice): Foundation architecture
- [YODAS2 Dataset](https://huggingface.co/datasets/espnet/yodas): Training data (~200,000 hours)
- [Qwen Team](https://huggingface.co/Qwen): Language model backbone
- [Facebook AudioSeal](https://huggingface.co/facebook/audioseal): Audio watermarking

### Special Thanks

- **Carlos Menke**: For invaluable efforts in gathering datasets and extensive benchmarking
- **AI Service Center Berlin-Brandenburg (KI-Servicezentrum)**: For providing GPU resources (8x H100)

## Citation

```bibtex
@software{kugelaudio2026,
  title = {KugelAudio: Open-Source Text-to-Speech for European Languages with Voice Cloning},
  author = {Kratzenstein, Kajo and Menke, Carlos},
  year = {2026},
  institution = {Hasso-Plattner-Institut},
  url = {https://huggingface.co/kugelaudio/kugelaudio-0-open}
}
```
