# ASR Inference

This directory contains inference modules for the LaCT ASR models.

## Recommended: CTC Decoder with KenLM

**Use `asr_ctc_decoder.py`** for best performance with:
- ✅ torchaudio's optimized CTC beam search decoder
- ✅ KenLM language model support
- ✅ Lexicon-constrained decoding
- ✅ Pretrained LibriSpeech decoder files

### Quick Start

```python
from inference.asr_ctc_decoder import ASRCTCInference

# Initialize with pretrained LibriSpeech decoder files
inference = ASRCTCInference(
    model_path="./checkpoints/best_model.pt",
    config_path="./configs/config.json",
    device="cuda",
    use_pretrained_librispeech=True,  # Auto-downloads LibriSpeech decoder files
    beam_size=1500,
    lm_weight=3.23,
    word_score=-0.26,
)

# Transcribe audio
result = inference.transcribe_file("audio.wav")
print(f"Transcription: {result['text']}")
print(f"N-best: {result['nbest']}")
```

### Command Line Usage

```bash
# Using pretrained LibriSpeech files (recommended)
python inference/asr_ctc_decoder.py \
    --model_path ./checkpoints/best_model.pt \
    --config_path ./configs/config.json \
    --use_pretrained_librispeech \
    --audio_file audio.wav \
    --beam_size 1500 \
    --lm_weight 3.23

# Using custom files
python inference/asr_ctc_decoder.py \
    --model_path ./checkpoints/best_model.pt \
    --config_path ./configs/config.json \
    --tokens tokens.txt \
    --lexicon lexicon.txt \
    --lm lm.bin \
    --audio_file audio.wav \
    --beam_size 1500
```

### Key Parameters

- **`beam_size`**: Beam width for decoding (default: 1500)
  - Larger = better quality, slower
  - Typical range: 500-2000
- **`lm_weight`**: Language model fusion weight (default: 3.23)
  - Higher = more LM influence
  - Typical range: 1.0-5.0
- **`word_score`**: Bonus for finishing words (default: -0.26)
  - Typical range: -1.0 to 0.0
- **`nbest`**: Number of hypotheses to return (default: 3)

### Evaluation

```bash
python inference/asr_ctc_decoder.py \
    --model_path ./checkpoints/best_model.pt \
    --config_path ./configs/config.json \
    --use_pretrained_librispeech \
    --mode evaluate \
    --test_data_dir /path/to/LibriSpeech \
    --test_subset test-clean \
    --beam_size 1500
```

## Deprecated: Custom Beam Search

`asr_inference.py` contains a custom beam search implementation that has been superseded by the torchaudio decoder. For new development, use `asr_ctc_decoder.py` instead.

## References

- **CTC Tutorial**: https://pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html
- **Wav2Letter Paper**: Connectionist temporal classification for labeling unsegmented sequence data
- **KenLM**: https://kheafield.com/code/kenlm/

