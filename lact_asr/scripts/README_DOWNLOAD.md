# LibriSpeech Download Guide

## Quick Start

The download scripts now use HuggingFace datasets by default (works in HPC environments where openslr.org may be blocked).

### Basic Usage

```bash
# Download minimal dataset (train-clean-100 + dev-clean)
./scripts/download_librispeech.sh

# Download to custom directory
./scripts/download_librispeech.sh --data-dir /path/to/data

# Download specific subsets
./scripts/download_librispeech.sh --subsets "train-clean-360 dev-clean"

# Download full dataset
./scripts/download_librispeech.sh --full
```

### Available Subsets

| Subset Name | Size | Hours | Description |
|------------|------|-------|-------------|
| `train-clean-100` | ~6 GB | 100h | Clean training speech |
| `train-clean-360` | ~23 GB | 360h | Clean training speech |
| `train-other-500` | ~30 GB | 500h | Other training speech |
| `dev-clean` | ~1 GB | 5h | Clean validation set |
| `dev-other` | ~1 GB | 5h | Other validation set |
| `test-clean` | ~1 GB | 5h | Clean test set |
| `test-other` | ~1 GB | 5h | Other test set |

## Sample Audio Files

The download script automatically:
1. **Validates** the downloaded dataset
2. **Extracts a sample** audio file from each subset
3. **Creates a waveform visualization** (if matplotlib is available)
4. **Saves the sample** to `{data_dir}/samples/` for inspection

### Sample Files Location

After download, you'll find:
```
/path/to/data/
├── LibriSpeech/
│   ├── train-clean-100/
│   ├── dev-clean/
│   └── ...
└── samples/
    ├── sample_train-clean-100.flac      # Audio file
    ├── sample_train-clean-100.txt       # Transcript + metadata
    └── sample_train-clean-100_waveform.png  # Visualization
```

### How to Listen to Samples

#### On a machine with audio output:

```bash
# Using ffplay (from ffmpeg)
ffplay -autoexit /path/to/data/samples/sample_train-clean-100.flac

# Using aplay (Linux)
aplay /path/to/data/samples/sample_train-clean-100.flac

# Using afplay (macOS)
afplay /path/to/data/samples/sample_train-clean-100.flac

# Using Python with sounddevice
python -c "import soundfile as sf; import sounddevice as sd; \
           data, sr = sf.read('/path/to/data/samples/sample_train-clean-100.flac'); \
           sd.play(data, sr); sd.wait()"
```

#### View the waveform visualization:

```bash
# Open the PNG file
open /path/to/data/samples/sample_train-clean-100_waveform.png  # macOS
xdg-open /path/to/data/samples/sample_train-clean-100_waveform.png  # Linux

# Or copy to your local machine
scp user@hpc:/path/to/data/samples/sample_*.png ./
```

## Python Script Direct Usage

For more control, use the Python script directly:

```bash
# Basic usage
python scripts/download_librispeech_hf.py \
    --data-dir /path/to/data \
    --subsets train.clean.100 dev.clean

# Skip validation (faster)
python scripts/download_librispeech_hf.py \
    --data-dir /path/to/data \
    --subsets train.clean.100 \
    --no-validate

# Don't save sample files
python scripts/download_librispeech_hf.py \
    --data-dir /path/to/data \
    --subsets train.clean.100 \
    --no-save-samples
```

## HuggingFace Subset Mapping

The script automatically maps openslr.org names to HuggingFace:

| Input Name | HuggingFace Config | HuggingFace Split | Output Name |
|------------|-------------------|-------------------|-------------|
| `train-clean-100` | `clean` | `train.100` | `train-clean-100` |
| `train-clean-360` | `clean` | `train.360` | `train-clean-360` |
| `train-other-500` | `other` | `train.500` | `train-other-500` |
| `dev-clean` | `clean` | `validation` | `dev-clean` |
| `dev-other` | `other` | `validation` | `dev-other` |
| `test-clean` | `clean` | `test` | `test-clean` |
| `test-other` | `other` | `test` | `test-other` |

Both formats work: `train-clean-100` or `train.clean.100`

## Validation Features

The validation process checks:
- ✓ Directory structure exists
- ✓ Audio files (.flac) are present
- ✓ Transcript files (.trans.txt) are present
- ✓ Sample audio can be loaded and decoded
- ✓ Sample transcript is readable
- ✓ Audio statistics (sample rate, duration, amplitude range)

Sample output:
```
Validating subset: train-clean-100
  ✓ Directory exists: /data/LibriSpeech/train-clean-100
  ✓ Audio files (.flac): 28539
  ✓ Transcript files (.trans.txt): 2484
  ✓ Sample audio valid: 103-1240-0000.flac (6.21s, 16000Hz)
  ✓ Sample transcript valid: 103-1240.trans.txt (16 utterances)

✓ Sample Audio Details:
  File: train-clean-100/103/1240/103-1240-0000.flac
  Sample rate: 16000 Hz
  Duration: 6.21 seconds
  Channels: 1
  Min amplitude: -0.4521
  Max amplitude: 0.3982
  Mean amplitude: 0.0003

✓ Transcript:
  Utterance ID: 103-1240-0000
  Text: CHAPTER ONE MISSUS RACHEL LYNDE IS SURPRISED
```

## Dependencies

Required Python packages:
```bash
pip install datasets soundfile tqdm numpy
```

Optional (for visualizations):
```bash
pip install matplotlib
```

## Troubleshooting

### Issue: Missing packages
```bash
pip install datasets soundfile tqdm numpy matplotlib
```

### Issue: HuggingFace cache fills up disk
Set cache directory:
```bash
export HF_DATASETS_CACHE=/path/to/large/storage/.cache
export HF_HOME=/path/to/large/storage/.cache
```

### Issue: Download is slow
HuggingFace streams large datasets. First download caches data for faster subsequent access.

### Issue: Unknown subset error
Make sure you're using valid subset names (see Available Subsets table above).

## Integration with Training

After download, use with training scripts:

```bash
# Using setup_and_train.sh (will skip download)
./scripts/setup_and_train.sh --skip-download \
    --data-dir /path/to/data

# Using train_librispeech.sh directly
./examples/train_librispeech.sh \
    --data-dir /path/to/data/LibriSpeech \
    --train-subset train-clean-100
```

## See Also

- `NETWORK_TROUBLESHOOTING.md` - Network issues and alternatives
- `setup_and_train.sh` - Complete pipeline script
- `train_librispeech.sh` - Training script

