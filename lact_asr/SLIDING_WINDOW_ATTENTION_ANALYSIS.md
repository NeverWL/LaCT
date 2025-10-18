# LaCT ASR Sliding Window Attention Analysis

## 🎯 Summary

**YES, the LaCT ASR model implements sliding window attention**, following the same pattern as the minimal causal LaCT implementation.

---

## ✅ Implementation Details

### 1. **Sliding Window Configuration**

All ASR configs define a `window_size` parameter:

```json
{
  "lact_chunk_size": 4096,
  "window_size": 4096,
  ...
}
```

From `configs/`:
- ✅ `base_asr_config.json` - window_size: 4096
- ✅ `improved_asr_config.json` - window_size: 4096  
- ✅ `regularized_asr_config.json` - window_size: 4096
- ✅ `large_asr_config.json` - window_size: 4096

### 2. **Flash Attention Integration**

In `layer_lact_asr.py` (lines 334-365), the model uses Flash Attention with window constraints:

```python
# Line 346-347: With padding mask
o = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
)

# Line 357-358: Without padding mask  
o = flash_attn_varlen_func(
    q.squeeze(0), k.squeeze(0), v.squeeze(0),
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
)

# Line 361-364: Standard path
o = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
)
```

**Key Points:**
- ✅ Window size is applied to all Flash Attention variants
- ✅ Format: `(self.window_size-1, 0)` = look back N-1 tokens, 0 forward (causal)
- ✅ Falls back to full attention if `window_size is None`

### 3. **Fallback Implementation**

For systems without Flash Attention (lines 367-400):

```python
if self.window_size is not None:
    # Sliding window attention
    mask = torch.ones(q_len, q_len, device=q.device, dtype=torch.bool)
    mask = torch.triu(mask, diagonal=1)  # Causal mask
    # Add window constraint
    for i in range(q_len):
        mask[i, :max(0, i - self.window_size + 1)] = True
    attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
else:
    # Full causal mask
    mask = torch.triu(torch.ones(q_len, q_len, device=q.device, dtype=torch.bool), diagonal=1)
    attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
```

**Key Points:**
- ✅ Implements sliding window via masking
- ✅ Each position can only attend to previous `window_size` tokens
- ✅ Maintains causality (no future tokens)

---

## 📊 Comparison with Minimal Implementation

### Minimal Implementation (`causal_lact_with_sliding_window_attn.py`)

```python
# Line 383-384
attn_output = flash_attn_func(
    attn_q, attn_k, attn_v,
    causal=True,
    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
)
```

### ASR Implementation (`layer_lact_asr.py`)

```python
# Line 361-365
o = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
)
```

**Result:** ✅ **IDENTICAL PATTERN**

---

## 🏗️ Architecture Overview

The LaCT ASR layer implements the **same architecture** as the minimal causal LaCT:

1. **Shared QKV Projection**
   ```python
   q, k, v = self.qkv(hidden_states_ds).chunk(3, dim=-1)
   ```

2. **Sliding Window Attention** (Lines 334-401)
   - Uses Flash Attention with window constraint
   - Falls back to manual windowed attention if needed
   - Window size: 4096 tokens (configurable)

3. **TTT (Test-Time Training)** (Lines 404-480)
   - Causal block-wise updates
   - Chunk size: 4096 tokens (matches window size)
   - Uses `block_causal_lact_swiglu` from `ttt_operation.py`

4. **Output Combination** (Line 388 in minimal, Line 503 in ASR)
   ```python
   # Minimal implementation
   output = attn_output + ttt_output
   
   # ASR implementation  
   o = o + ttt_x_normed  # Same pattern!
   ```

---

## 🎯 Key Differences from Minimal Implementation

### ASR-Specific Features:

1. **Temporal Downsampling** (Lines 278-286)
   ```python
   if self.audio_adapt and self.temporal_reduction > 1:
       hidden_states_ds = self.temporal_downsample(hidden_states_ds)
   ```
   - Reduces audio sequence length before processing
   - Helps with very long audio sequences

2. **Audio Feature Encoder** (Not in minimal impl)
   - Mel-spectrogram extraction
   - Convolutional feature encoding
   - BatchNorm layers

3. **CTC Head** (Not in minimal impl)
   - Projects to vocabulary size
   - Used for ASR loss computation

4. **RoPE (Rotary Position Embeddings)** (Lines 318, 429)
   - Applied to both attention and TTT
   - Better position encoding for long sequences

### Otherwise: **Core LaCT mechanism is IDENTICAL**

---

## ✅ Conclusion

The LaCT ASR implementation:

1. ✅ **Uses sliding window attention** (window_size = 4096)
2. ✅ **Follows the minimal causal LaCT pattern** exactly
3. ✅ **Combines windowed attention + TTT** in the same layer
4. ✅ **Shares QKV** between attention and TTT (GAU-style)
5. ✅ **Implements proper causality** for sequential processing

The only differences are **ASR-specific adaptations**:
- Audio feature extraction
- Temporal downsampling
- CTC output head
- Longer default window (4096 vs 2048)

**The core LaCT mechanism is implemented correctly according to the paper and minimal reference implementation!** 🎯

---

## 📝 Configuration Recommendations

Current settings (all configs):
```json
{
  "window_size": 4096,
  "lact_chunk_size": 4096
}
```

**Why these values?**
- ✅ Window = Chunk ensures full attention within each TTT update chunk
- ✅ 4096 tokens ≈ 25.6 seconds of audio at 160 hop_length
- ✅ Good balance between context and memory/computation

**Potential adjustments:**
- Shorter audio (< 10s): Could reduce to 2048
- Very long audio (> 1min): Could increase to 8192
- Memory constraints: Reduce both proportionally

---

**Verdict: LaCT ASR correctly implements sliding window attention as specified in the minimal implementation!**

