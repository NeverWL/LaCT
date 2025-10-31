#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH -J lact_asr_inference_test
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o lact_asr_inference_test.log

# Test inference on trained LaCT ASR model
# This script loads a trained checkpoint and tests transcription

# Load HPC modules
ml load gcc/12.2
ml load cuda/12.2

# Set cache directories
export HUGGING_FACE_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_DATASETS_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_HOME=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache

# Activate virtual environment
source /nfs/stak/users/limjar/hpc-share/myVenv/bin/activate

# Set environment variables
export HOME="/nfs/stak/users/limjar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /nfs/stak/users/limjar/hpc-share/LaCT/lact_asr

# Default configuration
DEFAULT_CHECKPOINT_DIR="./checkpoints/librispeech_base"
DEFAULT_DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech"
DEFAULT_TEST_SUBSET="train-clean-100"
DEFAULT_NUM_SAMPLES=10

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 [options]

Test inference on trained LaCT ASR model.

Options:
  -c, --checkpoint-dir DIR    Directory containing trained model (default: $DEFAULT_CHECKPOINT_DIR)
  -d, --data-dir DIR          Directory with LibriSpeech dataset (default: $DEFAULT_DATA_DIR)
  -s, --test-subset SUBSET    Test subset to use (default: $DEFAULT_TEST_SUBSET)
  -n, --num-samples NUM       Number of samples to test (default: $DEFAULT_NUM_SAMPLES)
  --beam-width WIDTH          Beam width for decoding (default: 1, greedy)
  -h, --help                  Show this help message

Examples:
  $0                                              # Test with defaults
  $0 -c ./checkpoints/my_model -n 20             # Test 20 samples from custom checkpoint
  $0 --test-subset dev-clean --beam-width 5      # Use dev set with beam search

EOF
}

# Parse arguments
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
DATA_DIR="$DEFAULT_DATA_DIR"
TEST_SUBSET="$DEFAULT_TEST_SUBSET"
NUM_SAMPLES="$DEFAULT_NUM_SAMPLES"
BEAM_WIDTH=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -s|--test-subset)
            TEST_SUBSET="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --beam-width)
            BEAM_WIDTH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

date
echo "🎤 Starting LaCT ASR Inference Test"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR Inference Test"
print_status "======================="
print_status "Checkpoint directory: $CHECKPOINT_DIR"
print_status "Data directory: $DATA_DIR"
print_status "Test subset: $TEST_SUBSET"
print_status "Number of samples: $NUM_SAMPLES"
print_status "Beam width: $BEAM_WIDTH"
echo ""

# Check if checkpoint exists
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    print_error "Checkpoint directory not found: $CHECKPOINT_DIR"
    print_status "Please train a model first using:"
    print_status "  sbatch scripts/setup_and_train.sh"
    exit 1
fi

# Check for model files
MODEL_FILE="$CHECKPOINT_DIR/checkpoint-step-58872.pt"
if [[ ! -f "$MODEL_FILE" ]]; then
    MODEL_FILE="$CHECKPOINT_DIR/latest_checkpoint.pt"
    if [[ ! -f "$MODEL_FILE" ]]; then
        print_error "No model checkpoint found in $CHECKPOINT_DIR"
        print_error "Expected: best_model.pt or latest_checkpoint.pt"
        exit 1
    fi
    print_warning "Using latest_checkpoint.pt (best_model.pt not found)"
fi

CONFIG_FILE="$CHECKPOINT_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_success "Found checkpoint: $MODEL_FILE"
print_success "Found config: $CONFIG_FILE"
echo ""

# Check if test data exists
TEST_DATA_PATH="$DATA_DIR/$TEST_SUBSET"
if [[ ! -d "$TEST_DATA_PATH" ]]; then
    print_error "Test data not found: $TEST_DATA_PATH"
    print_status "Available subsets in $DATA_DIR:"
    ls -1 "$DATA_DIR" 2>/dev/null || echo "  (directory not accessible)"
    exit 1
fi

print_success "Found test data: $TEST_DATA_PATH"
echo ""

# Run inference test
print_status "Running inference test..."
echo ""
echo "=================================================="

python << EOF
import sys
import torch
import torchaudio
from pathlib import Path
import json
import random

# Add parent directory to path
sys.path.append(str(Path.cwd()))

from lact_asr_model import LaCTASRConfig, LaCTASRForCTC
from data import LibriSpeechDataset

print("Loading model...")

# Load config
with open("$CONFIG_FILE", 'r') as f:
    config_dict = json.load(f)
config = LaCTASRConfig(**config_dict)

# Load model
checkpoint = torch.load("$MODEL_FILE", map_location='cuda')
model = LaCTASRForCTC(config)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to('cuda')
model.eval()

print(f"✓ Model loaded from $MODEL_FILE")
print(f"  Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
print(f"  Vocab size: {config.ctc_vocab_size}")

# Load training dataset first to get the correct vocabulary
print(f"\nLoading training vocabulary...")
train_dataset = LibriSpeechDataset(
    root_dir="$DATA_DIR",
    subset="train-clean-360",  # Use the same training set as the model
    sample_rate=config.sample_rate,
    max_duration=20.0,
    normalize_text=True,
)
print(f"✓ Training vocabulary loaded: {len(train_dataset.vocab)} characters")

# Load test dataset
print(f"\nLoading test dataset: $TEST_SUBSET")
test_dataset = LibriSpeechDataset(
    root_dir="$DATA_DIR",
    subset="$TEST_SUBSET",
    sample_rate=config.sample_rate,
    max_duration=20.0,
    normalize_text=True,
)

# CRITICAL: Use the same vocabulary as training
original_vocab_size = len(test_dataset.vocab)
test_dataset.vocab = train_dataset.vocab
test_dataset.char_to_idx = train_dataset.char_to_idx
test_dataset.idx_to_char = train_dataset.idx_to_char
print(f"✓ Using training vocabulary for $TEST_SUBSET")
print(f"  Original vocab size: {original_vocab_size} → Training vocab size: {len(train_dataset.vocab)}")

print(f"✓ Test dataset loaded: {len(test_dataset)} samples")

# Test on random samples
num_samples = min($NUM_SAMPLES, len(test_dataset))
print(f"\nTesting on {num_samples} random samples...")
print("=" * 60)

# Get random samples
indices = random.sample(range(len(test_dataset)), num_samples)

correct = 0
total = 0

for i, idx in enumerate(indices):
    sample = test_dataset[idx]
    
    # Prepare input
    audio = sample['audio'].unsqueeze(0).to('cuda')  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        outputs = model(audio_input=audio)
        logits = outputs.logits
        
        # Greedy decoding
        predictions = torch.argmax(logits, dim=-1)
        
        # Convert to text (simple greedy decode, remove blanks and duplicates)
        pred_indices = predictions[0].cpu().tolist()
        
        # Remove blanks (0) and consecutive duplicates
        decoded = []
        prev = None
        for idx in pred_indices:
            if idx != 0 and idx != prev:  # Not blank and not duplicate
                decoded.append(idx)
            prev = idx
        
        # Convert indices to characters
        pred_text = ''.join([test_dataset.idx_to_char.get(idx, '?') for idx in decoded])
        true_text = sample['text']
        
        # Simple word-level accuracy
        pred_words = pred_text.lower().split()
        true_words = true_text.lower().split()
        
        if pred_words == true_words:
            correct += 1
        total += 1
        
        print(f"\nSample {i+1}/{num_samples}:")
        print(f"  True: {true_text}")
        print(f"  Pred: {pred_text}")
        print(f"  Match: {'✓' if pred_words == true_words else '✗'}")

print("\n" + "=" * 60)
print(f"Results:")
print(f"  Exact matches: {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 60)

EOF

exit_code=$?

echo ""
echo "=================================================="
echo ""
date

if [[ $exit_code -eq 0 ]]; then
    print_success "Inference test completed successfully!"
    print_status "Model checkpoint: $MODEL_FILE"
    print_status "Test subset: $TEST_SUBSET"
    echo ""
    print_status "To run inference on your own audio files, use:"
    print_status "  python examples/inference_example.py"
    print_status "  (Edit the script to point to your audio files)"
else
    print_error "Inference test failed with exit code $exit_code"
    print_status "Check the output above for error details"
    exit 1
fi

