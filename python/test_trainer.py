#!/usr/bin/env python3
"""
Quick test script for the DeBleed trainer.
Generates synthetic test audio and runs a minimal training run.
"""

import os
import sys
import tempfile
import torch
import soundfile as sf
import numpy as np

# Test parameters
SAMPLE_RATE = 48000
DURATION = 5.0  # 5 seconds per file
NUM_CLEAN = 3
NUM_NOISE = 3


def generate_sine_wave(freq: float, duration: float, sr: int) -> torch.Tensor:
    """Generate a sine wave."""
    t = torch.linspace(0, duration, int(sr * duration))
    return torch.sin(2 * 3.14159 * freq * t)


def generate_noise(duration: float, sr: int) -> torch.Tensor:
    """Generate filtered noise."""
    samples = int(sr * duration)
    noise = torch.randn(samples) * 0.3
    return noise


def main():
    print("=== DeBleed Trainer Test ===")
    print()

    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        clean_dir = os.path.join(tmpdir, "clean")
        noise_dir = os.path.join(tmpdir, "noise")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(clean_dir)
        os.makedirs(noise_dir)
        os.makedirs(output_dir)

        print(f"Created temp dirs: {tmpdir}")
        print()

        # Generate test clean audio (sine waves at different frequencies)
        print("Generating clean audio files...")
        for i in range(NUM_CLEAN):
            freq = 220 * (i + 1)  # 220Hz, 440Hz, 660Hz
            audio = generate_sine_wave(freq, DURATION, SAMPLE_RATE)
            path = os.path.join(clean_dir, f"clean_{i}.wav")
            sf.write(path, audio.numpy(), SAMPLE_RATE)
            print(f"  Created {path}")

        # Generate test noise audio
        print("Generating noise audio files...")
        for i in range(NUM_NOISE):
            audio = generate_noise(DURATION, SAMPLE_RATE)
            path = os.path.join(noise_dir, f"noise_{i}.wav")
            sf.write(path, audio.numpy(), SAMPLE_RATE)
            print(f"  Created {path}")

        print()
        print("Running trainer with minimal settings...")
        print("=" * 50)

        # Import trainer module
        import trainer

        # Test audio file collection
        clean_files = trainer.collect_audio_files(clean_dir)
        noise_files = trainer.collect_audio_files(noise_dir)
        print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files")

        # Test audio loading
        sample_audio = trainer.load_audio_file(clean_files[0])
        print(f"Loaded audio shape: {sample_audio.shape}")
        print(f"Audio duration: {len(sample_audio) / SAMPLE_RATE:.2f}s")

        # Test STFT
        print()
        print("Testing STFT/iSTFT...")
        mag, phase = trainer.compute_stft(sample_audio)
        print(f"STFT output - Magnitude: {mag.shape}, Phase: {phase.shape}")

        reconstructed = trainer.compute_istft(mag, phase)
        print(f"iSTFT output: {reconstructed.shape}")

        # Verify reconstruction
        orig_len = min(len(sample_audio), reconstructed.shape[1])
        error = torch.mean(torch.abs(sample_audio[:orig_len] - reconstructed[0, :orig_len]))
        print(f"Reconstruction error: {error.item():.6f}")

        # Test dataset creation
        print()
        print("Creating dataset...")
        dataset = trainer.SyntheticMixtureDataset(
            clean_files=clean_files,
            noise_files=noise_files,
            samples_per_epoch=10  # Very small for testing
        )

        # Test getting a sample
        mixture_mag, clean_mag, ideal_mask = dataset[0]
        print(f"Mixture mag shape: {mixture_mag.shape}")
        print(f"Clean mag shape: {clean_mag.shape}")
        print(f"Ideal mask shape: {ideal_mask.shape}")
        print(f"Mask range: [{ideal_mask.min():.3f}, {ideal_mask.max():.3f}]")

        # Test model creation
        print()
        print("Creating model...")
        model = trainer.MaskEstimator(n_freq_bins=trainer.N_FREQ_BINS, hidden_dim=32)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

        # Test forward pass
        print()
        print("Testing forward pass...")
        test_input = mixture_mag.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            test_output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

        # Quick training test (1 epoch)
        print()
        print("Running 1 epoch of training...")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        trainer_obj = trainer.Trainer(model, dataloader, device='cpu')
        loss = trainer_obj.train_epoch(0, 1)
        print(f"Training loss: {loss:.6f}")

        # Test ONNX export
        print()
        print("Testing ONNX export...")
        onnx_path = trainer.export_to_onnx(model, output_dir)
        print(f"ONNX model saved to: {onnx_path}")

        # Verify ONNX file exists
        assert os.path.exists(onnx_path), "ONNX file not created!"
        file_size = os.path.getsize(onnx_path) / 1024
        print(f"ONNX file size: {file_size:.1f} KB")

        # Test metadata export
        print()
        print("Testing metadata export...")
        metadata_path = trainer.save_metadata(output_dir, model, {'losses': [loss]})
        print(f"Metadata saved to: {metadata_path}")

        # Load and print metadata
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Metadata contents:")
        for k, v in metadata.items():
            print(f"  {k}: {v}")

        # Verify ONNX can be loaded by onnxruntime
        print()
        print("Verifying ONNX with onnxruntime...")
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            print(f"Input name: {input_name}")
            print(f"Output name: {output_name}")

            input_shape = session.get_inputs()[0].shape
            print(f"Expected input shape: {input_shape}")

            # Run inference
            test_np = np.random.randn(1, trainer.N_FREQ_BINS, 64).astype(np.float32)
            result = session.run([output_name], {input_name: test_np})
            print(f"ONNX inference output shape: {result[0].shape}")
            print(f"ONNX output range: [{result[0].min():.3f}, {result[0].max():.3f}]")
            print("ONNX Runtime inference: SUCCESS")
        except ImportError:
            print("onnxruntime not installed, skipping verification")
            print("Install with: pip install onnxruntime")

        print()
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)


if __name__ == '__main__':
    main()
