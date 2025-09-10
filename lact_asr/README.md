# LaCT ASR: Large Chunk Test-Time Training for Automatic Speech Recognition

This repository implements LaCT (Large Chunk Test-Time Training) for Automatic Speech Recognition, extending the original LaCT approach from language modeling to audio processing tasks.

## Overview

LaCT ASR combines the efficiency of large chunk processing with test-time training to create a powerful ASR system that can:

- Handle long-form audio sequences efficiently (up to several hours)
- Adapt to speakers and acoustic conditions during inference
- Process audio in chunks of 2048-8192 frames (20-80 seconds)
- Achieve competitive performance with reduced computational overhead

## Architecture

The model consists of:

1. **Audio Feature Extractor**: Mel-spectrogram extraction with convolutional preprocessing
2. **LaCT Transformer Layers**: Modified transformer blocks with fast weight updates
3. **CTC Head**: Connectionist Temporal Classification for sequence-to-sequence learning

Key features:
- **Fast Weights**: SwiGLU-based learnable parameters that update during inference
- **Sliding Window Attention**: Efficient attention mechanism for long sequences  
- **Test-Time Training**: Model parameters adapt using current audio context
- **Mixed Precision**: Support for efficient training and inference

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd LaCT/lact_asr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Flash Linear Attention:
```bash
pip install flash-linear-attention
```

4. (Optional) Install Flash Attention for better performance:
```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

### Automated Setup (Recommended)

**Complete setup and training in one command:**

```bash
# Automated download and training
./scripts/setup_and_train.sh

# See all available options
./scripts/show_options.sh
```

This will automatically:
- Download LibriSpeech dataset (~7GB)
- Install missing dependencies  
- Start training with optimal settings
- Resume automatically if interrupted

### Manual Setup

**Download LibriSpeech dataset:**

```bash
# Download minimal dataset (100 hours)
./scripts/download_librispeech.sh

# Download to custom directory
./scripts/download_librispeech.sh --data-dir /custom/path

# Download larger dataset (360 hours)
./scripts/download_librispeech.sh --dataset-size standard
```

**Train on LibriSpeech dataset:**

```bash
./examples/train_librispeech.sh --data-dir /path/to/LibriSpeech
```

Train on custom dataset:

```bash
python training/train_asr.py \
    --dataset_type generic \
    --train_manifest /path/to/train_manifest.json \
    --val_manifest /path/to/val_manifest.json \
    --config_path configs/base_asr_config.json \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --max_epochs 20 \
    --mixed_precision
```

### Inference

Transcribe audio files:

```bash
# Single file
python inference/asr_inference.py \
    --mode transcribe \
    --model_path ./checkpoints/best_model.pt \
    --audio_file /path/to/audio.wav

# Batch transcription
python inference/asr_inference.py \
    --mode transcribe \
    --model_path ./checkpoints/best_model.pt \
    --audio_dir /path/to/audio/directory \
    --output_file transcriptions.json \
    --batch_size 8
```

### Evaluation

Evaluate model performance:

```bash
python inference/asr_inference.py \
    --mode evaluate \
    --model_path ./checkpoints/best_model.pt \
    --test_manifest /path/to/test_manifest.json \
    --batch_size 8
```

## Data Format

### Manifest Files

Training and evaluation data should be provided in JSON manifest format, with one JSON object per line:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "transcription text"}
{"audio_filepath": "/path/to/audio2.wav", "text": "another transcription"}
```

### Supported Datasets

- **LibriSpeech**: Automatic manifest creation from LibriSpeech directory structure
- **Mozilla Common Voice**: Automatic manifest creation from Common Voice TSV files
- **Custom datasets**: Generic manifest-based loading

## Configuration

Model configurations are stored in JSON files. Key parameters:

```json
{
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attn_heads": 12,
  "num_lact_heads": 4,
  "lact_chunk_size": 4096,
  "window_size": 4096,
  "sample_rate": 16000,
  "n_mels": 80,
  "ctc_vocab_size": 32,
  "use_muon": true,
  "use_momentum": true
}
```

### Available Configurations

- `configs/base_asr_config.json`: Base model (768 hidden size, 12 layers)
- `configs/large_asr_config.json`: Large model (1024 hidden size, 24 layers)

## Model Architecture Details

### LaCT Layer Adaptations for ASR

The ASR-specific modifications include:

1. **Audio-aware chunking**: Optimized chunk sizes for audio sequences
2. **Temporal downsampling**: Optional frame rate reduction for efficiency
3. **Mel-spectrogram integration**: Native support for audio feature extraction
4. **CTC-optimized decoding**: Specialized decoding for speech recognition

### Fast Weight Updates

The model uses SwiGLU-based fast weights that update during inference:

```
f(x) = w1 @ (silu(w0 @ x) * (w2 @ x))
```

These weights adapt to:
- Speaker characteristics
- Acoustic conditions  
- Background noise
- Recording quality

## Performance

Expected performance on common datasets:

| Dataset | Model Size | WER | CER | Notes |
|---------|------------|-----|-----|-------|
| LibriSpeech test-clean | Base (768d) | ~8-12% | ~3-5% | Estimated |
| LibriSpeech test-other | Base (768d) | ~20-25% | ~8-12% | Estimated |
| Common Voice | Base (768d) | ~15-20% | ~5-8% | Estimated |

*Note: Actual performance will depend on training data, hyperparameters, and compute resources.*

## Code Reuse from LaCT LLM

This implementation reuses approximately 80% of the code from the original `lact_llm` implementation:

### Directly Reused Components (90%+ code reuse):
- `ttt_operation.py`: Core test-time training operations
- `configuration_lact_swiglu.py`: Base configuration class
- Model architecture backbone from `modeling_lact.py`
- Fast weight initialization and updates

### Adapted Components (50-70% code reuse):
- `layer_lact_swiglu.py` → `layer_lact_asr.py`: Added audio-specific modifications
- Model blocks and attention mechanisms

### New Components (Audio-specific):
- `audio_features.py`: Mel-spectrogram extraction and preprocessing
- `asr_dataset.py`: Dataset loading for speech recognition
- CTC loss and decoding logic
- Audio-specific training and evaluation scripts

## Advanced Usage

### Custom Audio Preprocessing

```python
from lact_asr_model import AudioFeatureExtractor, SpecAugment

# Custom feature extractor
feature_extractor = AudioFeatureExtractor(
    config,
    input_type="mel"  # or "raw"
)

# Add data augmentation
spec_augment = SpecAugment(
    freq_mask_param=15,
    time_mask_param=35,
    prob=0.5
)
```

### Model Customization

```python
from lact_asr_model import LaCTASRConfig, LaCTASRForCTC

# Custom configuration
config = LaCTASRConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    lact_chunk_size=8192,  # Larger chunks for long audio
    use_muon=True,
    use_momentum=True
)

# Initialize model
model = LaCTASRForCTC(config)
```

### Streaming Inference

The model supports streaming inference for real-time applications:

```python
# Process audio in chunks
chunk_size = 4096  # samples
for chunk in audio_stream:
    result = inference.transcribe_audio(chunk)
    print(result['text'])
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or chunk size
2. **Slow training**: Enable mixed precision and increase num_workers
3. **Poor performance**: Check data quality and increase training epochs
4. **Import errors**: Ensure flash-linear-attention is installed

### Performance Optimization

- Use mixed precision training (`--mixed_precision`)
- Increase batch size and gradient accumulation steps
- Use multiple GPUs with `torch.nn.DataParallel`
- Enable Flash Attention for better memory efficiency

## Contributing

This implementation extends the original LaCT work for ASR applications. Contributions are welcome for:

- Additional dataset support
- Improved decoding algorithms
- Performance optimizations
- Evaluation metrics

## Citation

If you use this code, please cite the original LaCT paper:

```bibtex
@article{zhang2025test,
  title={Test-time training done right},
  author={Zhang, Tianyuan and Bi, Sai and Hong, Yicong and Zhang, Kai and Luan, Fujun and Yang, Songlin and Sunkavalli, Kalyan and Freeman, William T and Tan, Hao},
  journal={arXiv preprint arXiv:2505.23884},
  year={2025}
}
```

## License

This project follows the same license as the original LaCT repository.
