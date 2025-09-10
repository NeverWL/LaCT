# LaCT ASR Scripts

This directory contains utility scripts for setting up and running LaCT ASR training with the LibriSpeech dataset.

## Scripts Overview

### 1. `download_librispeech.sh`
Downloads LibriSpeech dataset in the correct format for LaCT ASR training.

**Features:**
- Downloads any combination of LibriSpeech subsets
- Verifies file integrity and dataset structure
- Supports resuming interrupted downloads
- Creates test scripts for validation
- Handles different dataset sizes (minimal, standard, full)

**Usage:**
```bash
# Download minimal dataset (recommended for testing)
./scripts/download_librispeech.sh

# Download to custom directory
./scripts/download_librispeech.sh --data-dir /data/LibriSpeech

# Download full dataset
./scripts/download_librispeech.sh --full

# Download specific subsets
./scripts/download_librispeech.sh --subsets "train-clean-360 dev-clean"
```

### 2. `setup_and_train.sh`
Complete pipeline that downloads LibriSpeech and starts training in one command.

**Features:**
- Automated dataset download and setup
- Dependency checking and installation
- Flexible training configuration
- Resume capability
- Support for different dataset sizes

**Usage:**
```bash
# Complete setup and training (minimal dataset)
./scripts/setup_and_train.sh

# Use larger dataset
./scripts/setup_and_train.sh --dataset-size standard

# Custom training parameters
./scripts/setup_and_train.sh \
    --dataset-size standard \
    --batch-size 16 \
    --epochs 30 \
    --learning-rate 1e-4

# Just download, don't train
./scripts/setup_and_train.sh --download-only

# Use existing dataset
./scripts/setup_and_train.sh --skip-download --data-dir /existing/LibriSpeech
```

## Dataset Sizes

| Size | Subsets | Training Hours | Download Size | Use Case |
|------|---------|----------------|---------------|----------|
| **minimal** | train-clean-100, dev-clean | ~100h | ~7GB | Testing, development |
| **standard** | train-clean-360, dev-clean | ~360h | ~24GB | Full training |
| **full** | All clean subsets | ~500h+ | ~30GB+ | Research, best performance |

## Quick Start

### Option 1: Complete Automated Setup
```bash
cd /Users/jaredlim/LaCT/lact_asr
./scripts/setup_and_train.sh
```

This will:
1. Download LibriSpeech (minimal dataset)
2. Install missing dependencies
3. Start training with default parameters

### Option 2: Manual Setup
```bash
# 1. Download dataset
./scripts/download_librispeech.sh --data-dir /tmp/LibriSpeech

# 2. Install dependencies
pip install -r requirements.txt
pip install flash-linear-attention

# 3. Start training
./examples/train_librispeech.sh --data-dir /tmp/LibriSpeech/LibriSpeech
```

### Option 3: HPC/Cluster Setup
For HPC environments with job schedulers:

```bash
# Download dataset (interactive node)
./scripts/download_librispeech.sh --data-dir /shared/datasets/LibriSpeech

# Submit training job
sbatch examples/train_librispeech.sh --data-dir /shared/datasets/LibriSpeech/LibriSpeech
```

## Resuming Training

Both training scripts automatically resume from the latest checkpoint:

```bash
# This will automatically resume if checkpoints exist
./scripts/setup_and_train.sh --skip-download --data-dir /existing/path
```

Or manually:
```bash
./examples/train_librispeech.sh \
    --data-dir /path/to/LibriSpeech \
    --output-dir ./checkpoints/existing_run
```

## Troubleshooting

### Common Issues

**1. Download failures:**
```bash
# Resume interrupted download
./scripts/download_librispeech.sh --force

# Check network connectivity
curl -I http://www.openslr.org/resources/12/train-clean-100.tar.gz
```

**2. Missing dependencies:**
```bash
# Install all dependencies
pip install -r requirements.txt
pip install flash-linear-attention

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Dataset structure issues:**
```bash
# Verify dataset structure
./path/to/dataset/test_librispeech.sh

# Expected structure:
LibriSpeech/
├── train-clean-100/
│   ├── 19/
│   │   ├── 198/
│   │   │   ├── 19-198.trans.txt
│   │   │   └── *.flac files
```

**4. Training crashes:**
```bash
# Check CUDA memory
nvidia-smi

# Reduce batch size
./scripts/setup_and_train.sh --batch-size 4

# Disable mixed precision
# (requires modifying train_asr.py)
```

### Performance Tips

**For faster downloads:**
```bash
# Install pv for progress bars
sudo apt-get install pv  # Ubuntu/Debian
brew install pv          # macOS

# Use wget instead of curl
sudo apt-get install wget
```

**For faster training:**
```bash
# Use larger batch size (if GPU memory allows)
./scripts/setup_and_train.sh --batch-size 16

# Use multiple GPUs (requires code modification)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable Flash Attention
pip install flash-attn --no-build-isolation
```

## File Outputs

### Download Script Outputs:
- `$DATA_DIR/LibriSpeech/` - Extracted dataset
- `$DATA_DIR/LibriSpeech/*_manifest.json` - Auto-generated manifests
- `$DATA_DIR/test_librispeech.sh` - Dataset verification script

### Training Outputs:
- `$OUTPUT_DIR/checkpoint-step-*.pt` - Regular checkpoints
- `$OUTPUT_DIR/best_model.pt` - Best validation model
- `$OUTPUT_DIR/latest_checkpoint.pt` - Most recent checkpoint
- `$OUTPUT_DIR/config.json` - Model configuration

## Advanced Usage

### Custom Dataset Subsets:
```bash
./scripts/download_librispeech.sh \
    --subsets "train-clean-100 train-other-500 dev-clean test-clean"
```

### Training with Custom Config:
```bash
# Create custom config
cp configs/base_asr_config.json configs/my_config.json
# Edit my_config.json...

# Train with custom config
./examples/train_librispeech.sh \
    --config configs/my_config.json \
    --data-dir /path/to/LibriSpeech
```

### Batch Processing:
```bash
# Download multiple dataset sizes
for size in minimal standard full; do
    ./scripts/download_librispeech.sh \
        --data-dir "/data/LibriSpeech_$size" \
        --${size}
done
```

## Integration with Other Tools

### Weights & Biases Logging:
```bash
# Install wandb
pip install wandb
wandb login

# Modify train_asr.py to add wandb logging
# (requires code changes)
```

### TensorBoard Monitoring:
```bash
# Start TensorBoard (if logging is enabled)
tensorboard --logdir ./checkpoints/librispeech_base/logs
```

### Model Evaluation:
```bash
# Evaluate trained model
python inference/asr_inference.py \
    --mode evaluate \
    --model_path ./checkpoints/librispeech_base/best_model.pt \
    --test_manifest /path/to/test_manifest.json
```

## Support

For issues and questions:
1. Check the main README.md in the project root
2. Verify your dataset structure matches the expected format
3. Ensure all dependencies are installed correctly
4. Check GPU memory and CUDA compatibility

## License

These scripts follow the same license as the main LaCT project.
