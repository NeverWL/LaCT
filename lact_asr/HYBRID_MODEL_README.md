# Hybrid LaCT + Wav2Vec2 ASR Model

## Overview

This directory contains the implementation of a **Hybrid LaCT + Wav2Vec2 ASR Model** with Test-Time Training (TTT) capabilities. The model integrates:

1. **Pretrained Wav2Vec2 Feature Encoder** (7-layer CNN, modified stride for 5x downsampling)
2. **16 LaCT Transformer Layers** with fast-weight adapters (SwiGLU MLP, MuOn optimizer)
3. **CTC Decoder** with optional beam search and KenLM language model
4. **Test-Time Training (TTT)** procedure for improved robustness

## Architecture

### Feature Encoder
- **Base**: Pretrained Wav2Vec2 7-layer CNN feature encoder
- **Modification**: Stride adjusted to achieve **5x downsampling** (vs default 20x)
- **Output**: ~800 tokens per 10s audio (higher temporal resolution than default wav2vec2)

### Transformer Encoder
- **Layers**: 16 LaCT transformer blocks (vs 12 in BASE)
- **Hidden Size**: 768
- **FFN Dimension**: 3072 (same as wav2vec2 BASE)
- **Attention Heads**: 8 (wav2vec2 BASE style) or 12 (LaCT style)
- **Fast-Weight Adapter**: 
  - Single-headed SwiGLU MLP
  - Hidden dimension: 1536
  - MuOn optimizer for fast weights
- **Regularization**:
  - Dropout: 0.1 (transformer + encoder output)
  - LayerDrop: 0.05 (BASE-style)

### Fast-Weight Configuration
- **Number of blocks**: 16 (matches transformer depth)
- **Optimizer**: MuOn
- **Fast-weight LR init**: softplus(const_lr_bias) = 0.01
- **Memory size**: ~84M parameters (~25-30% of total)

## Files

### Configuration
- `lact_asr_model/configuration_hybrid_asr.py`: Hybrid model configuration
- `configs/hybrid_asr_config.json`: Example configuration file

### Model Implementation
- `lact_asr_model/modeling_hybrid_asr.py`: Hybrid model architecture
- `lact_asr_model/wav2vec2_encoder.py`: Wav2Vec2 feature encoder wrapper

### Training & Inference
- `training/train_hybrid_asr.py`: Training script with separate LR for encoder/transformer
- `inference/hybrid_asr_inference.py`: Inference script with TTT support
- `inference/ttt_procedure.py`: Test-Time Training procedure implementation

## Training

### Setup

1. Install dependencies:
```bash
pip install transformers torchaudio jiwer
```

2. Download LibriSpeech dataset (960h):
```bash
# Follow LibriSpeech download instructions
```

### Training Command

```bash
python training/train_hybrid_asr.py \
    --config_path configs/hybrid_asr_config.json \
    --data_dir /path/to/librispeech \
    --train_subset train-clean-360 \
    --val_subset dev-clean \
    --test_subset test-clean \
    --output_dir ./checkpoints/hybrid \
    --batch_size 8 \
    --max_epochs 25 \
    --peak_lr 5e-4 \
    --warmup_steps 10000 \
    --encoder_lr_scale 0.1 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --enable_ttt
```

### Key Training Parameters

- **Peak LR**: 5e-4 (triangular schedule with 10k warmup)
- **Encoder LR**: 0.1 × peak LR (lightly fine-tuned encoder)
- **Transformer LR**: peak LR (normal training)
- **Batch Size**: Configurable (crop to 250k audio samples ≈ 15.6s)
- **Gradient Accumulation**: 2x (effective batch size = 2 × batch_size)

### Learning Rate Schedule

- **Warmup**: Linear warmup for 10k steps (10% → 100% of peak LR)
- **Decay**: Cosine annealing to 1% of peak LR

## Inference with TTT

### Basic Inference

```bash
python inference/hybrid_asr_inference.py \
    --checkpoint ./checkpoints/hybrid/best_model.pt \
    --audio_file path/to/audio.wav \
    --use_ttt \
    --beam_size 50
```

### Dataset Evaluation

```bash
python inference/hybrid_asr_inference.py \
    --checkpoint ./checkpoints/hybrid/best_model.pt \
    --data_dir /path/to/librispeech \
    --subset test-clean \
    --use_ttt \
    --beam_size 50 \
    --move_emission_to_cpu
```

### TTT Configuration

- **Loss Type**: `masked_prediction` (default) or `entropy`
- **Mask Probability**: 0.15 (for masked prediction)
- **TTT Steps**: 1-2 steps per utterance
- **Chunk Size**: Full utterance (up to ~15s) or configurable

## Test-Time Training (TTT) Procedure

The TTT procedure adapts the model's fast weights to each input utterance during inference:

1. **Forward Pass**: Run through frozen encoder + LaCT stack
2. **Self-Supervised Loss**: 
   - **Masked Prediction**: Predict masked tokens from context (similar to BERT)
   - **Entropy Regularization**: Minimize entropy to encourage confident predictions
3. **Fast Weight Updates**: Update fast weights (1-2 steps with MuOn)
4. **Rerun Inference**: Use adapted model for final transcription

### TTT Loss Types

#### Masked Prediction Loss
- Randomly masks 15% of tokens
- Predicts masked tokens from context
- MSE loss between predicted and original hidden states
- Encourages model to capture local context

#### Entropy Regularization Loss
- Minimizes entropy of output distribution
- Encourages confident predictions
- Useful for calibration and robustness

## Performance Expectations

Based on the architecture design:

- **Training**: Compatible with LibriSpeech 960h dataset
- **Inference**: 
  - With TTT: Improved robustness under distribution shifts (noise, accent, channel)
  - Without TTT: Standard inference (faster)
- **Memory**: ~84M parameters for fast weights (~25-30% of total)

## Configuration Options

### Encoder Settings
- `use_wav2vec2_encoder`: Enable/disable Wav2Vec2 encoder
- `wav2vec2_model_name`: HuggingFace model identifier
- `encoder_target_ds_factor`: Target downsampling factor (5x default)
- `freeze_encoder`: Freeze pretrained encoder weights
- `encoder_lr_scale`: Learning rate scale for encoder (0.1 default)

### Transformer Settings
- `num_hidden_layers`: Number of LaCT layers (16 default)
- `hidden_size`: Model dimension (768)
- `intermediate_size`: FFN dimension (3072)
- `num_attn_heads`: Attention heads (8)
- `num_lact_heads`: Fast-weight heads (4)

### TTT Settings
- `enable_ttt`: Enable Test-Time Training
- `ttt_loss_type`: Loss type (`masked_prediction` or `entropy`)
- `ttt_mask_prob`: Mask probability (0.15)
- `ttt_steps`: Number of TTT steps (1-2)

## Notes

1. **Dependencies**: Requires `transformers` library for Wav2Vec2 encoder
2. **Memory**: TTT increases inference time but improves robustness
3. **Compatibility**: Works with existing LaCT ASR infrastructure (datasets, evaluation scripts)
4. **Vocab Size**: Auto-detected from dataset (typically 29-32 for LibriSpeech)

## Troubleshooting

### Import Errors
If you see import errors for hybrid model classes:
- Ensure `transformers` is installed: `pip install transformers`
- Check that all files are in the correct directory structure

### CUDA Memory Errors
- Reduce batch size or use gradient accumulation
- Disable TTT during training (only enable during inference)
- Use mixed precision training (`--mixed_precision`)

### Slow Inference
- Disable TTT if speed is more important than robustness
- Reduce beam size (trade-off between accuracy and speed)
- Use GPU acceleration

## Citation

If you use this hybrid model, please cite:
- LaCT: [LaCT paper citation]
- Wav2Vec2: [Wav2Vec2 paper citation]

