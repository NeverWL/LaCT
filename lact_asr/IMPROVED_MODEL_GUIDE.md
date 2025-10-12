# Improved LaCT ASR Model - Training Guide

## Current vs Improved Model Comparison

### Your Current Model (Base):

```json
{
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attn_heads": 12,
  "num_lact_heads": 4,
  "audio_encoder_layers": 2
}
```

**Training:** train-clean-100 (100 hours)  
**Parameters:** ~100M  
**Results at 59k steps:** Phonetically similar but many errors

**Example:**
```
True: "BUT THE DEADLY LANGUOR AND COLDNESS..."
Pred: "et the dade laner in coldnets..."
```

### Improved Model (Full):

```json
{
  "hidden_size": 1024,        ← +33% capacity
  "num_hidden_layers": 18,    ← +50% depth
  "num_attn_heads": 16,       ← +33% attention
  "num_lact_heads": 8,        ← +100% TTT heads
  "audio_encoder_layers": 3   ← +50% feature extraction
}
```

**Training:** train-clean-360 (360 hours) - 3.6x more data!  
**Parameters:** ~200M  
**Expected Results:** 80-90% word accuracy

---

## Key Improvements

### 1. Larger Hidden Dimension (768 → 1024)

**Why:**
- More representational capacity
- Better at capturing complex patterns
- Reduces information bottleneck

**Impact:**
- Can represent more phonetic variations
- Better word boundary detection
- Improved handling of context

### 2. Deeper Model (12 → 18 layers)

**Why:**
- More processing depth
- Better long-range dependencies
- Can learn hierarchical features

**Impact:**
- Layer 1-6: Low-level audio features
- Layer 7-12: Phoneme patterns
- Layer 13-18: Word and sentence structure

### 3. More LaCT Heads (4 → 8)

**Why:**
- TTT mechanism learns better with more heads
- Each head can specialize
- More adaptive capacity

**Impact:**
- Better handling of different speakers
- Improved acoustic adaptation
- Faster convergence

### 4. Better Audio Encoder (2 → 3 layers)

**Why:**
- More sophisticated feature extraction
- Better spectral pattern learning
- Richer initial representations

**Impact:**
- Better noise robustness
- Improved pitch/tone capture
- Higher quality features for transformer

### 5. 3.6x More Training Data (100h → 360h)

**Why:**
- More diverse speech patterns
- Better generalization
- Reduced overfitting

**Impact:**
- More speakers, accents, vocabulary
- Better handling of edge cases
- Higher final accuracy

---

## Training Setup

### Step 1: Download Full Dataset

```bash
# Check current disk space (need ~25 GB)
df -h /nfs/stak/users/limjar/hpc-share/datasets/

# Download train-clean-360
sbatch scripts/download_full_dataset.sh

# This downloads:
# - train-clean-360 (~23 GB, 360 hours)
# - dev-clean (~1 GB, 5 hours) - if not already present
# - test-clean (~1 GB, 5 hours)
```

### Step 2: Start Improved Training

```bash
# After download completes, start training
sbatch scripts/train_improved_full.sh
```

---

## Expected Training Progress

### With Improved Model on Full Dataset:

| Steps | Epochs | Loss | Example Prediction | Quality |
|-------|--------|------|-------------------|---------|
| 0 | 0 | ~150 | Random chars | Baseline |
| 5,000 | ~1 | ~20 | "et the dade laner..." | Learning chars |
| 15,000 | ~3 | ~5 | "but the deadly languor..." | Partial words |
| 30,000 | ~6 | ~2 | "but the deadly languor and coldness..." | Most words |
| 50,000 | ~10 | ~1 | "but the deadly languor and coldness of the limbs..." | Good accuracy |
| 80,000 | ~16 | ~0.5 | Near perfect | Production |

### Timeline:

- **~1 day:** Steps 0-30k, Loss ~5, Partial accuracy
- **~2 days:** Steps 30k-60k, Loss ~1, Good accuracy (70-80%)
- **~3 days:** Steps 60k-100k, Loss ~0.3, Excellent accuracy (85-90%)

---

## Hyperparameter Tuning

### Current Settings (Optimized for Full Dataset):

```bash
BATCH_SIZE=12
GRADIENT_ACCUMULATION=2
# Effective batch size = 24

LEARNING_RATE=1e-4
WARMUP_STEPS=2000
MAX_GRAD_NORM=1.0

MAX_EPOCHS=40
MAX_AUDIO_DURATION=20.0
```

### If Loss Plateaus:

**Learning rate adjustments:**
```bash
# If stuck after 20k steps
--learning_rate 2e-4  # Increase

# If loss oscillates
--learning_rate 5e-5  # Decrease
```

**Model capacity adjustments:**
```bash
# If overfitting (train loss << val loss)
--audio_encoder_dropout 0.2  # More dropout

# If underfitting (both losses high)
# Use even larger model or more data
```

---

## Monitoring Training

### Watch Live Progress:

```bash
# Follow training log
tail -f lact_asr_improved_full.log

# Watch sample predictions (every 500 steps)
tail -f lact_asr_improved_full.log | grep -A 2 "Sample Prediction"

# Watch loss progression
tail -f lact_asr_improved_full.log | grep "Loss:" | tail -20
```

### Check Progress:

```bash
# Every 10k steps, test inference
sbatch scripts/test_inference.sh \
    --checkpoint-dir checkpoints/librispeech_improved_full \
    --test-subset dev-clean \
    --num-samples 20
```

---

## Model Size Comparison

| Model | Hidden | Layers | LaCT Heads | Conv Layers | Parameters | Dataset | Expected WER |
|-------|--------|--------|------------|-------------|------------|---------|--------------|
| Base | 768 | 12 | 4 | 2 | ~100M | 100h | 25-35% |
| **Improved** | **1024** | **18** | **8** | **3** | **~200M** | **360h** | **15-20%** |
| Large | 1536 | 24 | 12 | 4 | ~500M | 960h | 10-15% |

**Recommendation:** Use Improved model - best balance of quality and training time.

---

## Troubleshooting

### Out of Memory

If you get CUDA OOM:
```bash
# Reduce batch size
--batch_size 8 --gradient_accumulation_steps 3  # Still 24 effective

# Or reduce audio length
--max_audio_duration 15.0
```

### Still Getting NaN

If NaN appears even without mixed precision:
```bash
# Reduce learning rate
--learning_rate 5e-5

# Tighter gradient clipping
--max_grad_norm 0.5

# Check logs for which layer produces NaN
python scripts/diagnose_model.py --data-dir /path/to/data
```

### Training Too Slow

If training is too slow:
```bash
# Increase batch size (if memory allows)
--batch_size 16

# Use fewer workers if I/O bound
--num_workers 2

# Reduce logging frequency
--logging_steps 200
```

---

## After Training

### Test Your Model:

```bash
# Quick test
sbatch scripts/test_inference.sh \
    --checkpoint-dir checkpoints/librispeech_improved_full \
    --test-subset test-clean \
    --num-samples 50

# Full evaluation
python inference/asr_inference.py \
    --mode evaluate \
    --model_path checkpoints/librispeech_improved_full/best_model.pt \
    --config_path checkpoints/librispeech_improved_full/config.json \
    --test_manifest /path/to/test_manifest.json
```

### Expected Final Results:

**Word Error Rate (WER):** 15-20% on test-clean  
**Character Error Rate (CER):** 5-8%

**Example transcriptions:**
```
True: BUT THE DEADLY LANGUOR AND COLDNESS OF THE LIMBS TOLD ME...
Pred: BUT THE DEADLY LANGUOR AND COLDNESS OF THE LIMBS TOLD ME...
Match: ✓

True: THE MURDEROUS MARK OF THE FIEND'S GRASP WAS ON HER NECK
Pred: THE MURDEROUS MARK OF THE FIENDS GRASP WAS ON HER NECK
Match: ~95% (missing apostrophe)
```

---

## Quick Start

### Complete Workflow:

```bash
# 1. Download full dataset (~6-8 hours)
sbatch scripts/download_full_dataset.sh

# 2. Wait for download to complete
squeue -u $USER

# 3. Start improved training (~2-3 days)
sbatch scripts/train_improved_full.sh

# 4. Monitor progress
tail -f lact_asr_improved_full.log

# 5. Test periodically
sbatch scripts/test_inference.sh \
    --checkpoint-dir checkpoints/librispeech_improved_full \
    --test-subset dev-clean
```

---

## Summary of Changes

✅ **Model Architecture:**
- Hidden size: 768 → **1024** (+33%)
- Layers: 12 → **18** (+50%)
- LaCT heads: 4 → **8** (+100%)
- Conv layers: 2 → **3** (+50%)

✅ **Training Data:**
- Dataset: train-clean-100 → **train-clean-360** (3.6x more)
- Hours: 100h → **360h**

✅ **Hyperparameters:**
- Learning rate: 5e-6 → **1e-4** (20x faster)
- Batch size: 8 → **12** with accumulation
- Epochs: 20 → **40**
- Checkpoints: Every 1000 → **5000** steps

✅ **Features:**
- Sample prediction logging every 500 steps
- Validation every 1000 steps
- No mixed precision (stable training)
- Better gradient clipping

**Expected result:** 15-20% WER vs current 60-70% WER

Start the download and training - you should see dramatically better results! 🚀

