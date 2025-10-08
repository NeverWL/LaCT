# LaCT ASR Architecture & Data Flow

## Complete Pipeline Overview

```
Raw Audio → Mel-Spec → Conv Encoder → Positional Encoding → 12x LaCT Layers → CTC Head → Loss
[8, 236160] → [8, 80, 1476] → [8, 1476, 768] → [8, 1476, 768] → [8, 1476, 768] → [8, 1476, 32] → scalar
```

---

## Detailed Shape Transformations

### 1. Input: Raw Audio Waveform
```
Shape: [batch_size, num_samples]
Example: [8, 236160]

Details:
- 8 audio samples in the batch
- 236160 samples = 14.76 seconds at 16kHz
- Values typically in range [-1.0, 1.0]
```

### 2. Mel-Spectrogram Extraction
```python
# AudioFeatureExtractor → MelSpectrogramExtractor
Input:  [8, 236160]
Output: [8, 80, 1476]
        [batch, n_mels, time_frames]

Parameters:
- sample_rate: 16000 Hz
- hop_length: 160 samples (10ms stride)
- win_length: 400 samples (25ms window)
- n_fft: 512
- n_mels: 80

Calculation:
time_frames = (num_samples - win_length) / hop_length + 1
            = (236160 - 400) / 160 + 1
            ≈ 1476 frames

Each frame = 10ms of audio
Total duration = 1476 × 10ms = 14.76 seconds ✓
```

### 3. Convolutional Encoder (2 layers)
```python
# AudioFeatureExtractor → conv_layers

Layer 1:
  Conv1d(80 → 160, kernel=3, stride=1, padding=1)
  + BatchNorm1d(160)
  + ReLU()
  + Dropout(0.1)
  
  Input:  [8, 80, 1476]
  Output: [8, 160, 1476]

Layer 2:
  Conv1d(160 → 768, kernel=3, stride=1, padding=1)
  + BatchNorm1d(768)
  + ReLU()
  + Dropout(0.1)
  
  Input:  [8, 160, 1476]
  Output: [8, 768, 1476]

After transpose:
  [8, 768, 1476] → [8, 1476, 768]
  [batch, hidden_size, time] → [batch, time, hidden_size]
```

**Key Point:** `stride=1, padding=1` means **time dimension preserved!**

### 4. Positional Encoding
```python
# AudioPositionalEncoding

Input:  [8, 1476, 768]
Output: [8, 1476, 768]

Adds sinusoidal position embeddings:
pos_encoding[pos, i] = sin(pos / 10000^(2i/d)) or cos(...)
```

### 5. LaCT Transformer Layers (12 layers)
```python
# LaCTASRModel → 12x LaCTASRBlock

Each block:
  1. RMSNorm
  2. LaCTASRLayer (Attention + TTT)
     - Sliding window attention (window_size=4096)
     - Fast weight updates (TTT mechanism)
     - num_attn_heads=12, num_lact_heads=4
  3. RMSNorm
  4. MLP (feed-forward)
     - hidden_size → intermediate_size → hidden_size
     - SwiGLU activation

Input:  [8, 1476, 768]
Output: [8, 1476, 768]  # Same shape through all 12 layers
```

**Key Point:** Sequence length **never changes** in transformer layers!

### 6. CTC Head (Output Projection)
```python
# LaCTASRForCTC → ctc_head

Linear(768 → 32, bias=False)

Input:  [8, 1476, 768]
Output: [8, 1476, 32]
        [batch, time_steps, vocab_size]

Then log_softmax:
log_probs = torch.log_softmax(logits, dim=-1)
Output: [8, 1476, 32]
```

Each time step predicts a distribution over 32 characters.

### 7. CTC Loss
```python
# Transpose for CTC: [batch, time, vocab] → [time, batch, vocab]
log_probs_transposed = [1476, 8, 32]

CTCLoss(
    predictions: [1476, 8, 32],  # 1476 time steps per sample
    targets: [8, 231],           # Up to 231 characters per sample
    input_lengths: [8],          # Actual lengths: [1463, 1248, ...]
    label_lengths: [8]           # Actual lengths: [231, 159, ...]
)

Output: scalar loss value
```

---

## Input/Output Length Relationship

### Audio Length → Feature Length Conversion
```
audio_samples = 236160
hop_length = 160

feature_frames = (audio_samples + hop_length - 1) // hop_length
               = (236160 + 159) // 160
               = 1477 frames

# This is why you see 1477 in logits shape!
```

### CTC Constraint
```
For CTC to work:
  input_length >= label_length

Example from your log:
  input_lengths: [1463, 1248, 1291, 1476, 512, 1445, 741, 1277]
  label_lengths: [231,  159,  185,  163,  77,  209,  112, 183]
  
  1463 >= 231 ✓
  1248 >= 159 ✓
  512 >= 77 ✓
  All valid!
```

---

## Memory Footprint

### Approximate memory usage per batch:

```
Audio input:        8 × 236160 × 4 bytes = 7.5 MB
Mel-spectrogram:    8 × 80 × 1476 × 4 bytes = 3.8 MB
After conv:         8 × 768 × 1476 × 4 bytes = 36 MB
Hidden states:      8 × 1476 × 768 × 4 bytes = 36 MB per layer
                    × 12 layers × 2 (forward + backward) = 864 MB
Gradients:          ~same as parameters
Optimizer states:   ~2x parameters (Adam)

Total per batch: ~2-3 GB with batch_size=8
```

---

## Where NaN Can Occur

Based on your logs showing NaN in logits:

### Most Likely Locations:

1. **LaCT Layer Initialization** (line ~195 in layer_lact_asr.py)
   ```python
   # Fast weight initialization
   self.w0_init = torch.randn(...) * fw_init_gain / sqrt(d_h)
   # If d_h is 0 or fw_init_gain is inf → NaN
   ```

2. **RMSNorm** (used before each attention/MLP)
   ```python
   # If variance is 0 → division by zero
   variance = hidden_states.pow(2).mean(-1, keepdim=True)
   hidden_states = hidden_states / torch.sqrt(variance + eps)
   ```

3. **Attention Scores** (in fallback attention)
   ```python
   # If scale is 0 or inf
   scale = 1.0 / math.sqrt(head_dim)
   attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
   ```

4. **TTT Learning Rate** (line ~206 in layer_lact_asr.py)
   ```python
   # If base_lr_inv is too large
   lr = F.softplus(lr_proj + self.base_lr_inv)
   # softplus(very_large_number) = inf
   ```

---

## Diagnostic Script Output

When you run `sbatch scripts/diagnose_model.sh`, you'll see:

```
✓ Config created
✓ Model created on device: cuda:0
✓ Total parameters: 123456789
✓ No NaN/Inf in model parameters (or ❌ Found NaN in X parameters)
✓ Dummy input created
✓ Forward pass completed
  Loss: 234.5678
  Logits shape: torch.Size([2, 100, 32])
  Logits range: [-12.3456, 8.9012]
✅ Model appears healthy!
```

Or if there's an issue:
```
❌ Found NaN in 5 parameters:
     - model.layers.0.attn.w0_init
     - model.layers.0.attn.w2_init
     ...
```

This will pinpoint exactly which parameter or layer is causing the NaN! 🎯

---

## Submit the Diagnostic

```bash
# Submit to SLURM
sbatch scripts/diagnose_model.sh

# Check status
squeue -u $USER

# View results
tail -f lact_asr_diagnose.log
```

The diagnostic will complete in ~1-2 minutes and tell us exactly where the NaN originates!
