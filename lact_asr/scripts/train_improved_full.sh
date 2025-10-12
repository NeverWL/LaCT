#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH -J lact_asr_improved_full
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH -o lact_asr_improved_full.log
#SBATCH --constraint=h200

# Training script for IMPROVED LaCT ASR model on FULL LibriSpeech dataset
# This uses a larger model (1024 hidden, 18 layers) on train-clean-360

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

# Configuration
DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech"
OUTPUT_DIR="./checkpoints/librispeech_improved_full"
CONFIG_PATH="./configs/improved_asr_config.json"
TRAIN_SUBSET="train-clean-360"  # Full 360 hours
VAL_SUBSET="dev-clean"

# Training hyperparameters
BATCH_SIZE=12  # Larger model needs slightly smaller batch
MAX_EPOCHS=40
LEARNING_RATE="1e-4"
WARMUP_STEPS=2000
MAX_GRAD_NORM=1.0
GRADIENT_ACCUMULATION=2  # Effective batch size = 24
MAX_AUDIO_DURATION=20.0
NUM_WORKERS=4

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

date
echo "🚀 Starting IMPROVED LaCT ASR Training on Full Dataset"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR Improved Full Training"
print_status "================================"
print_status "Configuration:"
print_status "  Model config: $CONFIG_PATH"
print_status "  Model: 1024 hidden, 18 layers, 8 LaCT heads"
print_status "  Data directory: $DATA_DIR"
print_status "  Training subset: $TRAIN_SUBSET (360 hours)"
print_status "  Validation subset: $VAL_SUBSET"
print_status "  Output directory: $OUTPUT_DIR"
print_status ""
print_status "Training hyperparameters:"
print_status "  Batch size: $BATCH_SIZE (x${GRADIENT_ACCUMULATION} accumulation = effective $((BATCH_SIZE * GRADIENT_ACCUMULATION)))"
print_status "  Learning rate: $LEARNING_RATE"
print_status "  Max epochs: $MAX_EPOCHS"
print_status "  Warmup steps: $WARMUP_STEPS"
print_status "  Gradient clipping: $MAX_GRAD_NORM"
print_status "  Mixed precision: DISABLED (prevents NaN)"
print_status "  Max audio duration: $MAX_AUDIO_DURATION seconds"
echo ""

# Check if dataset exists
if [[ ! -d "$DATA_DIR/$TRAIN_SUBSET" ]]; then
    print_error "Training subset not found: $DATA_DIR/$TRAIN_SUBSET"
    print_status "Available subsets:"
    ls -1 "$DATA_DIR" 2>/dev/null || echo "  (cannot list directory)"
    print_status ""
    print_status "To download train-clean-360:"
    print_status "  ./scripts/download_librispeech.sh \\"
    print_status "    --data-dir $(dirname $DATA_DIR) \\"
    print_status "    --subsets train-clean-360"
    exit 1
fi

print_success "Dataset found: $DATA_DIR/$TRAIN_SUBSET"
echo ""

# Check if config exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    print_error "Config file not found: $CONFIG_PATH"
    exit 1
fi

print_success "Config found: $CONFIG_PATH"
echo ""

# Check for existing checkpoints
if [[ -f "$OUTPUT_DIR/latest_checkpoint.pt" ]]; then
    print_warning "Found existing checkpoint in $OUTPUT_DIR"
    print_status "Training will resume from latest checkpoint"
    RESUME_FLAG="--resume_from_checkpoint $OUTPUT_DIR/latest_checkpoint.pt"
else
    print_status "Starting fresh training (no existing checkpoint)"
    RESUME_FLAG=""
fi
echo ""

# Build training command
print_status "Building training command..."

TRAIN_CMD="python training/train_asr.py"
TRAIN_CMD="$TRAIN_CMD --config_path $CONFIG_PATH"
TRAIN_CMD="$TRAIN_CMD --dataset_type librispeech"
TRAIN_CMD="$TRAIN_CMD --data_dir $DATA_DIR"
TRAIN_CMD="$TRAIN_CMD --train_subset $TRAIN_SUBSET"
TRAIN_CMD="$TRAIN_CMD --val_subset $VAL_SUBSET"
TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --max_epochs $MAX_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
TRAIN_CMD="$TRAIN_CMD --max_grad_norm $MAX_GRAD_NORM"
TRAIN_CMD="$TRAIN_CMD --max_audio_duration $MAX_AUDIO_DURATION"
TRAIN_CMD="$TRAIN_CMD --num_workers $NUM_WORKERS"
TRAIN_CMD="$TRAIN_CMD --device cuda"
TRAIN_CMD="$TRAIN_CMD --logging_steps 100"
TRAIN_CMD="$TRAIN_CMD --save_steps 5000"
TRAIN_CMD="$TRAIN_CMD --eval_steps 1000"
TRAIN_CMD="$TRAIN_CMD $RESUME_FLAG"
# Note: NO --mixed_precision flag (disabled by default)

echo "=================================================="
print_status "Training Command:"
echo "$TRAIN_CMD"
echo "=================================================="
echo ""

print_status "Estimated training time: 2-3 days for full convergence"
print_status "Checkpoints saved every 5000 steps to: $OUTPUT_DIR"
print_status "Monitor progress: tail -f $PWD/lact_asr_improved_full.log"
echo ""

# Execute training
print_status "Starting training..."
echo ""

eval $TRAIN_CMD

exit_code=$?

echo ""
date

if [[ $exit_code -eq 0 ]]; then
    echo "✅ Training completed successfully!"
    print_status "Model checkpoints saved to: $OUTPUT_DIR"
    print_status "Best model: $OUTPUT_DIR/best_model.pt"
    print_status "Latest checkpoint: $OUTPUT_DIR/latest_checkpoint.pt"
    echo ""
    print_status "To test the model:"
    print_status "  sbatch scripts/test_inference.sh \\"
    print_status "    --checkpoint-dir $OUTPUT_DIR \\"
    print_status "    --test-subset dev-clean \\"
    print_status "    --num-samples 50"
else
    print_error "Training failed with exit code $exit_code"
    print_status "Check the log file for errors: lact_asr_improved_full.log"
    exit 1
fi

