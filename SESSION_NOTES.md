# DeBleed Training Session Notes - Jan 4, 2025

## Current Status
Mac Mini unreachable (went to sleep or network issue). Training may need restart.

## Optimization Attempts (Council Consulted)

### What Worked
- DataLoader optimizations (num_workers, pin_memory) - code added but needs testing

### What Didn't Work on MPS
1. **torch.compile**: Hangs during first forward pass on MPS
2. **AMP (float16)**: Causes Metal shader compilation errors:
   ```
   Failed to stich function with error: XPC_ERROR_CONNECTION_INVALID
   ```
3. **num_workers > 0**: May cause silent crashes with MPS

### Current Trainer Configuration
- torch.compile: **Disabled** on MPS (experimental, causes hangs)
- AMP: **Disabled** on MPS (Metal shader issues)
- num_workers: **0** (safer with MPS)
- Training runs in hybrid mode: Neural network on MPS, filters on CPU

## Training Audio Locations (Mac Mini)
- **Clean**: `~/Documents/Millie for training/` (294 files)
- **Noise**: `~/Documents/Stage Noise for Training/` (2 files)

## Training Command
```bash
cd ~/Downloads/DeBleedInstaller/DeBleed.app/Contents/Resources/python && \
nohup python3 neural5045_trainer.py \
  --clean_audio_dir ~/Documents/'Millie for training' \
  --noise_audio_dir ~/Documents/'Stage Noise for Training' \
  --output_path ~/Library/DeBleed/Models/T45_MM_optimized \
  --epochs 150 --batch_size 16 --samples_per_epoch 2000 \
  > ~/Library/DeBleed/Models/T45_optimized_stdout.log 2>&1 &
```

## Original Performance (baseline)
- ~5s per batch, 125 batches per epoch
- ~10.4 min per epoch
- **~26 hours total for 150 epochs**

## Future Optimization Ideas
1. Try different batch sizes on fresh restart
2. Profile the training loop to identify actual bottlenecks
3. Consider training on CUDA (cloud GPU) if MPS continues to have issues
4. Wait for PyTorch MPS improvements (torch.compile, AMP)

## SSH Command
```bash
ssh ksellarsm4lt@100.100.165.179
```

## This Machine
- M4 Max, 36GB RAM
- Main dev machine at `/Users/ksellarsm4lt/Documents/DeBleed/`
