# KugelAudio Node - Stuttering Fix Guide

## 🚨 IMPORTANT: Install with Editable Mode First!

**Before trying any fixes below, make sure kugelaudio-open is installed in editable mode!**

### Why Editable Mode Matters

The `kugelaudio-open` package must be installed with the `-e` (editable) flag so code changes take effect. Without this, editing files won't do anything!

**Windows Portable Users:**

```bash
cd ComfyUI/custom_nodes/ComfyUI-KugelAudio
..\..\..\python_embeded\python.exe -m pip install --no-deps --force-reinstall -e ./kugelaudio-open
```

**Or use the provided batch file:**

Double-click **`reinstall_no-deps.bat`** in the ComfyUI-KugelAudio folder.

**What this does:**
- `--no-deps`: Won't touch your existing dependencies (safe!)
- `--force-reinstall`: Reinstalls the package
- `-e` (editable): Creates a link to the source folder so code changes take effect after restart

**After running this, restart ComfyUI completely!**

---

## Version Compatibility Fix

If you're experiencing audio stuttering or micro-freezes after the editable install, this is a version compatibility issue with your Python/PyTorch environment.

---

## For CUDA Users (Windows/Linux with NVIDIA GPU)

### 🔍 Check Your Versions First

Run these commands:

```bash
# Check Python version
python --version

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

### ⚠️ The Problem

**If you see:**
- Python version: `3.13.x` ❌
- PyTorch version: `2.9.1+cu130` or `2.9.1+cu131` ❌

**This causes stuttering because:**
- Python 3.13 has audio buffer allocation bugs
- PyTorch CUDA 13.0+ uses different kernels that break audio continuity

### ✅ Required Versions (Working Configuration)

| Component | Required Version |
|-----------|-----------------|
| **Python** | 3.10.x, 3.11.x, or 3.12.x (NOT 3.13) |
| **PyTorch** | 2.9.0+cu128 or 2.9.1+cu128 (NOT cu130/cu131) |

These are the versions specified by the [original KugelAudio repository](https://github.com/Kugelaudio/kugelaudio-open).

### 🛠️ Fix Instructions

#### Option 1: Reinstall ComfyUI with Correct Python (Recommended)

1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. Uninstall current ComfyUI (or create new environment)
3. Reinstall ComfyUI with Python 3.12
4. Install PyTorch cu128:
   ```bash
   pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
   ```

#### Option 2: Downgrade PyTorch Only (If You Have Python 3.10-3.12)

**If your Python version is already 3.10, 3.11, or 3.12:**

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

**Then restart ComfyUI completely.**

### ✨ After Fixing

Once you match these exact versions:
1. Restart ComfyUI
2. Generate test audio
3. Stuttering should be completely gone

### 📚 Why These Versions?

- **Python 3.13**: Changed memory allocation → audio buffer discontinuities ([transformers issue #35443](https://github.com/huggingface/transformers/issues/35443))
- **CUDA 13.0**: Different cuDNN kernels → audio concatenation timing issues
- **cu128**: Tested and validated by KugelAudio developers

### ❓ Still Having Issues?

Report with:
- Output of `python --version`
- Output of `python -c "import torch; print(torch.__version__)"`
- Your GPU model

---

## For MPS Users (macOS with Apple Silicon)

### 🔍 Check Your Versions First

Run these commands:

```bash
# Check Python version
python --version

# Check PyTorch version
python -c "import torch; print(torch.__version__)"
```

### ⚠️ The Problem

**MPS (Metal Performance Shaders) backend has known compatibility issues with this model architecture.**

Symptoms:
- Audio stuttering/micro-freezes
- Possible `mps_matmul` errors
- Inconsistent generation quality

### ✅ Solution: Use CPU Mode

The CPU backend is **stable and tested** - it's slower but produces smooth, stutter-free audio.

### 🛠️ Fix Instructions

#### Step 1: Force CPU Mode in the Node

1. Open your KugelAudio node in ComfyUI
2. Find the **`device`** dropdown setting
3. Change it from `auto` or `mps` to **`cpu`**
4. Generate audio - stuttering should be gone

#### Step 2 (Optional): Downgrade PyTorch

**If you're using PyTorch 2.10.0 or newer:**

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

Then restart ComfyUI.

### 📊 Expected Results

| Mode | Speed | Quality | Stuttering |
|------|-------|---------|------------|
| MPS | Faster | ❌ Issues | ✅ Yes |
| CPU | Slower | ✅ Perfect | ❌ No |

**Note:** CPU mode uses more processing time but ensures stable, smooth audio output.

### ✅ Recommended Versions

| Component | Required Version |
|-----------|-----------------|
| **Python** | 3.10.x, 3.11.x, or 3.12.x (NOT 3.13) |
| **PyTorch** | 2.9.0 or 2.9.1 |
| **Device Mode** | CPU (not MPS) |

### 📚 Why MPS Doesn't Work?

- MPS has limited compatibility with the model's attention mechanisms
- Audio processing operations behave differently on MPS vs CUDA/CPU
- The original KugelAudio repository doesn't officially support MPS

### ✨ After Fixing

Once you switch to CPU mode:
1. Restart ComfyUI
2. Set device to `cpu` in node settings
3. Generate test audio
4. Stuttering should be completely gone

### ❓ Still Having Issues?

Report with:
- Output of `python --version`
- Output of `python -c "import torch; print(torch.__version__)"`
- Confirmation that device is set to `cpu`

---

## 📚 Reference

Original repository requirements: [KugelAudio Prerequisites](https://github.com/Kugelaudio/kugelaudio-open#prerequisites)
