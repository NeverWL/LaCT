# Network Troubleshooting for LibriSpeech Download

If you're experiencing connection timeouts or network issues when downloading LibriSpeech, this guide provides alternative solutions.

## Problem: Connection Timeout

```
Connecting to www.openslr.org (www.openslr.org)|136.243.171.4|:80... failed: Connection timed out.
```

This typically occurs in HPC environments with strict firewall rules that block direct internet access.

---

## Solution 1: Use HuggingFace Datasets (Recommended for HPC)

HuggingFace datasets are often accessible even when direct HTTP downloads are blocked.

### Setup

```bash
# Install required packages
pip install datasets soundfile tqdm

# Set cache directory (important for HPC)
export HF_DATASETS_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_HOME=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
```

### Download

```bash
# Minimal dataset (train-clean-100 + dev-clean)
python scripts/download_librispeech_hf.py \
    --data-dir /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT \
    --subsets train.clean.100 dev.clean

# Standard dataset (train-clean-360 + dev-clean)
python scripts/download_librispeech_hf.py \
    --data-dir /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT \
    --subsets train.clean.360 dev.clean

# Full dataset
python scripts/download_librispeech_hf.py \
    --data-dir /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT \
    --subsets train.clean.100 train.clean.360 dev.clean test.clean
```

### Then start training with:

```bash
./examples/train_librispeech.sh \
    --data-dir /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech \
    --train-subset train-clean-100
```

---

## Solution 2: Manual Download + Upload

If HuggingFace is also blocked, download on your local machine and upload to HPC.

### On your local machine:

```bash
# Download LibriSpeech
cd /tmp
wget http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget http://www.openslr.org/resources/12/dev-clean.tar.gz
```

### Upload to HPC:

```bash
# Using scp
scp train-clean-100.tar.gz limjar@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/
scp dev-clean.tar.gz limjar@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/

# Or using rsync
rsync -avz --progress train-clean-100.tar.gz \
    limjar@submit.hpc.engr.oregonstate.edu:/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/
```

### On HPC, extract:

```bash
cd /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT
tar -xzf train-clean-100.tar.gz
tar -xzf dev-clean.tar.gz
```

---

## Solution 3: Request Network Access from HPC Admin

Contact your HPC administrator to whitelist:
- `www.openslr.org` (IP: 136.243.171.4)
- Or enable HTTP/HTTPS access for your jobs

---

## Solution 4: Use Proxy (if available)

If your HPC provides a proxy server:

```bash
export http_proxy=http://proxy.your-institution.edu:8080
export https_proxy=http://proxy.your-institution.edu:8080
export HTTP_PROXY=http://proxy.your-institution.edu:8080
export HTTPS_PROXY=http://proxy.your-institution.edu:8080

# Then run download script
./scripts/download_librispeech.sh --data-dir /path/to/data
```

---

## Verification

After downloading (using any method), verify the dataset:

```bash
# Check directory structure
ls -lh /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech/

# Should see:
# train-clean-100/
# dev-clean/

# Check audio files
find /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech/train-clean-100 -name "*.flac" | head -5

# Check transcripts
find /nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech/train-clean-100 -name "*.trans.txt" | head -5
```

---

## Quick Reference: Dataset Sizes

| Subset | Size | Hours | Files |
|--------|------|-------|-------|
| train-clean-100 | ~6 GB | 100h | ~28,000 |
| train-clean-360 | ~23 GB | 360h | ~104,000 |
| train-other-500 | ~30 GB | 500h | ~148,000 |
| dev-clean | ~1 GB | 5h | ~2,700 |
| dev-other | ~1 GB | 5h | ~2,900 |
| test-clean | ~1 GB | 5h | ~2,600 |
| test-other | ~1 GB | 5h | ~2,900 |

---

## Still Having Issues?

1. Check available disk space: `df -h /nfs/stak/users/limjar/hpc-share/datasets/`
2. Check write permissions: `touch /nfs/stak/users/limjar/hpc-share/datasets/test.txt && rm /nfs/stak/users/limjar/hpc-share/datasets/test.txt`
3. Check your SLURM logs for detailed error messages
4. Try running the download script interactively on a compute node to see detailed errors

For more help, see the main README.md or open an issue on GitHub.

