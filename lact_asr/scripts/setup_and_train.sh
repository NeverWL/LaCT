#!/bin/bash
#SBATCH -t 0-24:00:00
#SBATCH -J lact_asr_setup_train
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=125G
#SBATCH -o lact_asr_setup_train.log

# Complete setup and training script for LaCT ASR with LibriSpeech
# This script downloads LibriSpeech and starts training in one go

# Load HPC modules
ml load gcc/12.2
ml load cuda/12.2

# Set cache directories
export HUGGING_FACE_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_DATASETS_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_HOME=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache

# Activate virtual environment
source /nfs/stak/users/limjar/hpc-share/myVenv/bin/activate

# Set environment variables for memory optimization
export HOME="/nfs/stak/users/limjar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /nfs/stak/users/limjar/hpc-share/LaCT/lact_asr

# Default configuration
DEFAULT_DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT"
DEFAULT_CHECKPOINT_DIR="./checkpoints/librispeech_base"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

show_usage() {
    cat << EOF
Usage: $0 [options]

Complete setup and training pipeline for LaCT ASR with LibriSpeech.
This script will:
1. Download LibriSpeech dataset (if not already present)
2. Start LaCT ASR training

Options:
  -d, --data-dir DIR          Directory for LibriSpeech dataset (default: $DEFAULT_DATA_DIR)
  -o, --output-dir DIR        Directory for training checkpoints (default: $DEFAULT_CHECKPOINT_DIR)
  -s, --dataset-size SIZE     Dataset size: minimal, standard, or full (default: minimal)
                              - minimal: train-clean-100 + dev-clean (~100 hours)
                              - standard: train-clean-360 + dev-clean (~360 hours)  
                              - full: all clean subsets (~500+ hours)
  --skip-download            Skip dataset download (assume already downloaded)
  --download-only            Only download dataset, don't start training
  -b, --batch-size SIZE      Training batch size (default: 8)
  -e, --epochs NUM           Maximum training epochs (default: 20)
  -l, --learning-rate RATE   Learning rate (default: 5e-6)
  --mixed-precision          Enable mixed precision training (default: disabled due to NaN issues)
  --resume                   Resume from existing checkpoint if available
  -h, --help                 Show this help message

Examples:
  $0                                    # Download minimal dataset and start training
  $0 --dataset-size standard           # Use larger training set (360h)
  $0 --skip-download -d /existing/path # Use existing LibriSpeech dataset
  $0 --download-only                   # Just download, don't train
  $0 --resume                          # Resume training from checkpoint

EOF
}

# Parse arguments
DATA_DIR="$DEFAULT_DATA_DIR"
OUTPUT_DIR="$DEFAULT_CHECKPOINT_DIR"
DATASET_SIZE="minimal"
SKIP_DOWNLOAD=false
DOWNLOAD_ONLY=false
BATCH_SIZE=16
MAX_EPOCHS=30
LEARNING_RATE="1e-4"
MIXED_PRECISION=false
RESUME=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--dataset-size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        -l|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --mixed-precision)
            MIXED_PRECISION=false
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate dataset size
case $DATASET_SIZE in
    minimal)
        DOWNLOAD_SUBSETS="train-clean-100 dev-clean"
        TRAIN_SUBSET="train-clean-100"
        ;;
    standard)
        DOWNLOAD_SUBSETS="train-clean-360 dev-clean"
        TRAIN_SUBSET="train-clean-360"
        ;;
    full)
        DOWNLOAD_SUBSETS="train-clean-100 train-clean-360 dev-clean test-clean"
        TRAIN_SUBSET="train-clean-360"
        ;;
    *)
        echo "Invalid dataset size: $DATASET_SIZE (must be minimal, standard, or full)"
        exit 1
        ;;
esac

# Get script directory
SCRIPT_DIR="$(pwd)/scripts"
PROJECT_DIR="$(pwd)"

date
echo "🚀 Starting LaCT ASR Setup and Training Pipeline"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR Setup and Training Pipeline"
print_status "====================================="
print_status "Data directory: $DATA_DIR"
print_status "Output directory: $OUTPUT_DIR"  
print_status "Dataset size: $DATASET_SIZE ($DOWNLOAD_SUBSETS)"
print_status "Training subset: $TRAIN_SUBSET"
print_status "Skip download: $SKIP_DOWNLOAD"
print_status "Download only: $DOWNLOAD_ONLY"
if [[ "$DOWNLOAD_ONLY" == "false" ]]; then
    print_status "Batch size: $BATCH_SIZE"
    print_status "Max epochs: $MAX_EPOCHS"
    print_status "Learning rate: $LEARNING_RATE"
    print_status "Mixed precision: $MIXED_PRECISION"
    print_status "Resume training: $RESUME"
fi
echo ""

# Step 1: Download LibriSpeech dataset
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    print_status "Step 1: Downloading LibriSpeech dataset..."
    
    # Check if dataset already exists
    LIBRISPEECH_PATH="$DATA_DIR/LibriSpeech"
    if [[ -d "$LIBRISPEECH_PATH" && "$RESUME" == "false" ]]; then
        print_warning "Dataset already exists at $LIBRISPEECH_PATH"
        print_status "Using existing dataset (SLURM non-interactive mode)"
        FORCE_DOWNLOAD=""
    else
        FORCE_DOWNLOAD=""
    fi
    
    # Run download script
    "$SCRIPT_DIR/download_librispeech.sh" \
        --data-dir "$DATA_DIR" \
        --subsets "$DOWNLOAD_SUBSETS" \
        $FORCE_DOWNLOAD
    
    if [[ $? -ne 0 ]]; then
        echo "Dataset download failed!"
        exit 1
    fi
    
    print_success "Dataset download completed"
else
    print_status "Skipping dataset download"
    LIBRISPEECH_PATH="$DATA_DIR/LibriSpeech"
    if [[ ! -d "$LIBRISPEECH_PATH" ]]; then
        echo "Error: LibriSpeech dataset not found at $LIBRISPEECH_PATH"
        echo "Either provide correct path or remove --skip-download flag"
        exit 1
    fi
fi

# Exit if download-only mode
if [[ "$DOWNLOAD_ONLY" == "true" ]]; then
    print_success "Dataset download completed. Training skipped (--download-only mode)"
    print_status "To start training, run:"
    print_status "  $PROJECT_DIR/examples/train_librispeech.sh --data-dir $LIBRISPEECH_PATH"
    exit 0
fi

# Step 2: Check dependencies
print_status "Step 2: Checking training dependencies..."

# Ensure we're in the project directory (already changed above)
echo "Current directory: $(pwd)"

# Virtual environment should be activated by now
if [[ -n "$VIRTUAL_ENV" ]]; then
    print_success "Virtual environment active: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected - this may cause issues"
fi

# Check key Python packages
python_deps=("torch" "torchaudio" "transformers")
missing_deps=()

for dep in "${python_deps[@]}"; do
    if ! python -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    print_warning "Missing Python dependencies: ${missing_deps[*]}"
    print_status "Installing missing dependencies..."
    pip install -r requirements.txt
    
    # Try installing flash-linear-attention
    if ! python -c "import flash_linear_attention" 2>/dev/null; then
        print_status "Installing flash-linear-attention..."
        pip install flash-linear-attention || print_warning "Failed to install flash-linear-attention (optional)"
    fi
fi

print_success "Dependencies check completed"

# Step 3: Start training
print_status "Step 3: Starting LaCT ASR training..."

# Build training command
TRAIN_CMD="$PROJECT_DIR/examples/train_librispeech.sh"
TRAIN_CMD="$TRAIN_CMD --data-dir $LIBRISPEECH_PATH"
TRAIN_CMD="$TRAIN_CMD --output-dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --train-subset $TRAIN_SUBSET"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --epochs $MAX_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"

if [[ "$MIXED_PRECISION" == "true" ]]; then
    TRAIN_CMD="$TRAIN_CMD --mixed_precision"
    print_warning "Mixed precision enabled - may cause NaN issues"
fi

echo "🎯 Training command:"
echo "  $TRAIN_CMD"
echo ""

# Execute training
eval $TRAIN_CMD

echo ""
date
if [[ $? -eq 0 ]]; then
    echo "✅ LaCT ASR setup and training completed successfully!"
    print_status "Model checkpoints saved to: $OUTPUT_DIR"
    print_status "Best model: $OUTPUT_DIR/best_model.pt"
    print_status "Latest checkpoint: $OUTPUT_DIR/latest_checkpoint.pt"
    echo ""
    print_status "To resume training later, run:"
    print_status "  $TRAIN_CMD"
    print_status "(The script will automatically detect and resume from the latest checkpoint)"
else
    echo "❌ Training failed with exit code $?"
    exit 1
fi
