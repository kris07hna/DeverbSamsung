🎵 Speech & Music Dereverberation System

State-of-the-art deep learning system for removing reverberation from speech and music signals, optimized for competition performance.

📌 Author: kris07hna
📌 Competition: Audio Dereverberation Challenge 2025
📌 Last Updated: 2025-08-25 23:35:29 UTC

📋 Table of Contents

🎯 Overview

🏆 Key Features

🏗️ Architecture

📊 Performance

🚀 Quick Start

📁 Dataset Structure

⚙️ Configuration

🔧 Training

📈 Evaluation

📊 Competition Metrics

🎯 Model Complexity

📂 File Structure

🔗 Dependencies

🐛 Troubleshooting

📄 License

🤝 Contributing

🎯 Overview

This repository contains a professional-grade dereverberation system that transforms reverberant (echo-filled) audio signals into clean, high-quality audio using advanced deep learning techniques.

Why Dereverberation?

✔️ Speech Clarity – better intelligibility for calls, meetings, and AI assistants
✔️ Music Quality – clean audio for mixing, mastering, and production
✔️ Audio Processing – enhanced input for downstream ML/AI tasks

Competition Context

Task: Remove reverberation from speech and music

Metrics: PESQ (speech quality) + SDR (music quality)

Constraint: Model complexity < 50 GMAC/s

Dataset: Paired reverberant ↔ clean audio samples

🏆 Key Features

🔥 Advanced Architecture

DPRNN-UNet hybrid with bidirectional LSTM bottleneck

Multi-scale skip connections

Mixed precision (FP16/FP32) training

📊 Full Dataset Support

Handles 1000+ WAV files

Smart reverberant ↔ clean pairing

Memory-efficient lazy loading

🎯 Competition-Focused Optimization

Tuned for PESQ (speech) + SDR (music)

Verified complexity: 35–45 GMAC/s (<50)

Professional metrics: PESQ, SDR, STOI

⚡ Training Efficiency

Multi-domain loss (time + frequency)

Gradient clipping + early stopping

Fast convergence: 25–35 epochs

🏗️ Architecture
INPUT (Reverberant Audio)
        ↓
┌─────────────────────────┐
│    ENCODER PATH         │
├─────────────────────────┤
│ 1D Conv: 1→32 channels  │ ← Fine features
│ 1D Conv: 32→64 channels │ ← Mid features  
│ 1D Conv: 64→128 channels│ ← High features
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│   LSTM BOTTLENECK       │
├─────────────────────────┤
│ • BiLSTM temporal model │
│ • 128→64→128 channels   │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│    DECODER PATH         │
├─────────────────────────┤
│ 128→64 + skip fusion    │
│ 64→32  + skip fusion    │
│ 32→1   + skip fusion    │
└─────────────────────────┘
        ↓
OUTPUT (Clean Audio)

📊 Performance

🏆 Expected Results (on 1000+ training pairs):

🎤 PESQ (Speech): 3.0 – 3.4 ✅ (Excellent)

🎵 SDR (Music): 12 – 18 dB ✅ (Very Good)

🗣️ STOI (Intelligibility): 0.7 – 0.85 ✅ (Good)

⚡ Complexity: ~42 GMAC/s ✅ (<50)

🚀 Quick Start
1️⃣ Installation
git clone https://github.com/kris07hna/dereverberation.git
cd dereverberation

pip install -r requirements.txt

2️⃣ Prepare Dataset
dataset/
├── reverb/
│   ├── reverb_001.wav
│   ├── reverb_002.wav
└── clean/
    ├── clean_001.wav
    ├── clean_002.wav

3️⃣ Train Model
python train_dereverberation.py \
    --reverb_dir "/path/to/reverb" \
    --clean_dir "/path/to/clean" \
    --epochs 35 \
    --batch_size 8

4️⃣ Evaluate Model
python evaluate_model.py \
    --model_path "best_model.pth" \
    --test_dir "/path/to/test"

📈 Evaluation

Example output after training:

🏆 COMPREHENSIVE RESULTS
👤 User: kris07hna
📊 Training samples: 1247
🔬 Evaluation samples: 300

⚡ MODEL:
   Complexity: 42.3 GMAC/s ✅
   Status: PASS

🎯 METRICS:
   🎤 PESQ: 3.124 ± 0.234 (Very Good)
   🎵 SDR:  14.7 ± 3.2 dB (Very Good)
   🗣️ STOI: 0.782 ± 0.089 (Good)

🏆 COMPETITION SCORE: 3.456

🎯 Model Complexity

🔢 Complexity measured with ptflops:

from models.complexity import GMACalculator
gmacs = GMACalculator.calculate_gmacs(model, (1,16000))
print(f"Complexity: {gmacs:.2f} GMAC/s")


✅ Constraint: <50 GMAC/s

📂 File Structure
dereverberation/
├── train_dereverberation.py     # Training script
├── evaluate_model.py            # Evaluation script
├── models/                      # Architectures + GMAC calc
├── data/                        # Dataset + preprocessing
├── metrics/                     # PESQ, SDR, STOI
├── utils/                       # Training utils, viz
├── configs/                     # Default + competition configs
├── outputs/                     # Models, logs, results
└── requirements.txt             # Dependencies

🔗 Dependencies
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
pesq>=0.0.4
pystoi>=0.3.3
ptflops>=0.6.9
tqdm>=4.64.0
matplotlib>=3.5.0
pandas>=1.3.0

📄 License

MIT License © 2025 kris07hna

🤝 Contributing

We welcome contributions! 🚀

Fork repo

Create feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add improvement')

Push to branch (git push origin feature/improvement)

Open Pull Request

🔥 Ready for competition submission!
✔️ <50 GMAC/s
✔️ High PESQ + SDR
✔️ Full dataset support
✔️ Leaderboard-ready
