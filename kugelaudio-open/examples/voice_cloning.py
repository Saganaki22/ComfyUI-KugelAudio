#!/usr/bin/env python3
"""Example of voice cloning with KugelAudio using a voice prompt.

All generated audio is automatically watermarked for identification,
which is especially important for voice cloning to prevent misuse.
"""

import torch
from kugelaudio_open import (
    KugelAudioForConditionalGenerationInference,
    KugelAudioProcessor,
)


def main():
    # Configuration
    model_id = "kugelaudio/kugelaudio-0-open"
    voice_prompt_path = "path/to/voice_prompt.wav"  # Replace with actual path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {model_id}...")

    # Load model and processor
    model = KugelAudioForConditionalGenerationInference.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    processor = KugelAudioProcessor.from_pretrained(model_id)

    # Text to synthesize with the cloned voice
    text = "This speech is generated using a voice prompt to clone a specific speaker's voice characteristics."

    print(f"Generating speech for: '{text}'")

    # Process input with voice prompt (pass the path directly to the processor)
    inputs = processor(
        text=text,
        voice_prompt=voice_prompt_path,  # Path to reference audio
        return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate speech (watermark is automatically applied)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            cfg_scale=3.0,
            max_new_tokens=2048,
        )

    audio = outputs.speech_outputs[0]

    # Save output
    output_path = "cloned_voice_output.wav"
    processor.save_audio(audio, output_path)
    print(f"Audio saved to {output_path}")


if __name__ == "__main__":
    main()
