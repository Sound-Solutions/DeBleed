DeBleed Neural Trainer
======================

SETUP (run once):
  1. Open Terminal
  2. cd to this folder
  3. Run: chmod +x install.sh train.sh
  4. Run: ./install.sh

TRAINING:
  1. Put your audio folders in this directory:
     - "Millie for training" (clean vocals)
     - "Stage Noise for Training" (bleed/noise)

  2. Run: ./train.sh "Millie for training" "Stage Noise for Training" 50
     (50 = number of epochs, adjust as needed)

  3. Wait for training to complete (can take 1-2+ hours)

  4. Model will be saved to: ./trained_model/neural5045.onnx

FOLDER STRUCTURE:
  DeBleedTrainer/
  ├── install.sh              <- Run first
  ├── train.sh                <- Run to train
  ├── neural5045_trainer.py   <- Training code
  ├── differentiable_svf.py   <- Filter code
  ├── Millie for training/    <- Your clean audio
  ├── Stage Noise for Training/ <- Your noise audio
  └── trained_model/          <- Output (created automatically)
