#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH -J librispeech_download_full
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --mem=32G
#SBATCH -o librispeech_download_full.log

# Download full LibriSpeech dataset for improved training
# Downloads train-clean-100 + train-clean-360 + train-other-500 + dev-clean + test-clean

# Load HPC modules
ml load gcc/12.2
ml load cuda/12.2

# Set cache directories
export HUGGING_FACE_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_DATASETS_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_HOME=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache

# Activate virtual environment
source /nfs/stak/users/limjar/hpc-share/myVenv/bin/activate

# Change to project directory
cd /nfs/stak/users/limjar/hpc-share/LaCT/lact_asr

# Configuration
DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT"
SUBSETS="train-clean-100 train-clean-360 train-other-500 dev-clean test-clean"

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}[INFO]${NC} Downloading Full LibriSpeech Dataset"
echo -e "${BLUE}[INFO]${NC} ======================================"
echo -e "${BLUE}[INFO]${NC} Data directory: $DATA_DIR"
echo -e "${BLUE}[INFO]${NC} Subsets: $SUBSETS"
echo -e "${BLUE}[INFO]${NC} Estimated size: ~25 GB"
echo ""

date

# Run download script
./scripts/download_librispeech.sh \
    --data-dir "$DATA_DIR" \
    --subsets "$SUBSETS"

exit_code=$?

echo ""
date

if [[ $exit_code -eq 0 ]]; then
    echo -e "${GREEN}[SUCCESS]${NC} Full dataset download completed!"
    echo ""
    echo -e "${BLUE}[INFO]${NC} Dataset location: $DATA_DIR/LibriSpeech"
    echo -e "${BLUE}[INFO]${NC} To start training with improved model:"
    echo -e "${BLUE}[INFO]${NC}   sbatch scripts/train_improved_full.sh"
else
    echo "Dataset download failed with exit code $exit_code"
    exit 1
fi

