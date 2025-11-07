#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -J lact_asr_regularized
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH -o lact_asr_regularized.log

# Training script for REGULARIZED LaCT ASR model
# Smaller model (640d, 12 layers) with stronger regularization to prevent overfitting

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

# Configuration - REGULARIZED TO PREVENT OVERFITTING
DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech"
OUTPUT_DIR="./checkpoints/librispeech_regularized"
CONFIG_PATH="./configs/regularized_asr_config.json"
TRAIN_SUBSET="train-clean-360"  # Full 360 hours
VAL_SUBSET="dev-clean"          # Keep same validation set
TEST_SUBSET="test-clean"        # For monitoring only (not model selection)

# Training hyperparameters - OPTIMIZED FOR REGULARIZATION
BATCH_SIZE=20  # Larger batch for smaller model
MAX_EPOCHS=25  # Fewer epochs to prevent overfitting
LEARNING_RATE="3e-5"  # Lower learning rate for stability
WARMUP_STEPS=2500  # Longer warmup
MAX_GRAD_NORM=0.5  # Stricter gradient clipping
GRADIENT_ACCUMULATION=2  # Effective batch size = 40
MAX_AUDIO_DURATION=25.0  # Slightly longer audio
NUM_WORKERS=4
TEST_EVAL_STEPS=3000  # Evaluate on test-clean every 3000 steps

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
echo "🎯 Starting REGULARIZED LaCT ASR Training (Anti-Overfitting)"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR REGULARIZED Training Configuration"
print_status "==========================================="
print_status "🎯 ANTI-OVERFITTING MEASURES:"
print_status "  ✓ Smaller model: 640d hidden, 12 layers (vs 1024d, 18 layers)"
print_status "  ✓ Stronger regularization: 0.25 dropout (vs 0.15)"
print_status "  ✓ Lower learning rate: 3e-5 (vs 1e-4)"
print_status "  ✓ Stricter gradient clipping: 0.5 (vs 1.0)"
print_status "  ✓ Fewer epochs: 25 (vs 40) to prevent overfitting"
print_status "  ✓ Larger effective batch size: 40 (vs 24)"
print_status ""
print_status "Model architecture:"
print_status "  Hidden size: 640 (d_model)"
print_status "  Layers: 12 (L)"
print_status "  Attention heads: 8"
print_status "  FFN size: 2560 (d_ff)"
print_status "  LaCT heads: 4"
print_status "  Estimated parameters: ~150M (vs 311M)"
print_status ""
print_status "Training data:"
print_status "  Training subset: $TRAIN_SUBSET (360 hours)"
print_status "  Validation subset: $VAL_SUBSET"
print_status "  Test subset: $TEST_SUBSET (monitoring only)"
print_status "  Data directory: $DATA_DIR"
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
print_status "  Test evaluation: every $TEST_EVAL_STEPS steps"
echo ""

# Check if dataset exists
if [[ ! -d "$DATA_DIR/$TRAIN_SUBSET" ]]; then
    print_error "Training subset not found: $DATA_DIR/$TRAIN_SUBSET"
    exit 1
fi

if [[ ! -d "$DATA_DIR/$VAL_SUBSET" ]]; then
    print_error "Validation subset not found: $DATA_DIR/$VAL_SUBSET"
    exit 1
fi

if [[ ! -d "$DATA_DIR/$TEST_SUBSET" ]]; then
    print_error "Test subset not found: $DATA_DIR/$TEST_SUBSET"
    exit 1
fi

print_success "Training dataset found: $DATA_DIR/$TRAIN_SUBSET"
print_success "Validation dataset found: $DATA_DIR/$VAL_SUBSET"
print_success "Test dataset found: $DATA_DIR/$TEST_SUBSET"
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
TRAIN_CMD="$TRAIN_CMD --test_subset $TEST_SUBSET"
TRAIN_CMD="$TRAIN_CMD --test_eval_steps $TEST_EVAL_STEPS"
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
TRAIN_CMD="$TRAIN_CMD --save_steps 3000"  # Save more frequently
TRAIN_CMD="$TRAIN_CMD --eval_steps 1000"  # Regular evaluation
TRAIN_CMD="$TRAIN_CMD $RESUME_FLAG"

echo "=================================================="
print_status "Training Command:"
echo "$TRAIN_CMD"
echo "=================================================="
echo ""

print_status "🎯 EXPECTED IMPROVEMENTS:"
print_status "  ✓ Reduced overfitting (smaller model + stronger regularization)"
print_status "  ✓ Better generalization to test-clean"
print_status "  ✓ More stable training (lower learning rate)"
print_status "  ✓ Faster training and inference (fewer parameters)"
print_status "  ✓ Better parameter-to-data ratio"
print_status "  ✓ Real-time monitoring of test-clean performance"
print_status ""
print_status "Training strategy:"
print_status "  ✓ Stop early if validation loss plateaus"
print_status "  ✓ Monitor for signs of overfitting"
print_status "  ✓ Focus on validation performance, not training loss"
print_status "  ✓ Test-clean evaluation every $TEST_EVAL_STEPS steps (monitoring only)"
print_status "  ✓ Model selection based on dev-clean, NOT test-clean"
print_status ""
print_status "Estimated training time: 12-18 hours (vs 2-3 days for larger model)"
print_status "Monitor progress: tail -f $PWD/lact_asr_regularized.log"
echo ""

# Execute training
print_status "Starting regularized training..."
echo ""

eval $TRAIN_CMD

exit_code=$?

echo ""
date

if [[ $exit_code -eq 0 ]]; then
    echo "✅ Regularized training completed successfully!"
    print_status "Model checkpoints saved to: $OUTPUT_DIR"
    print_status "Best model: $OUTPUT_DIR/best_model.pt"
    print_status "Latest checkpoint: $OUTPUT_DIR/latest_checkpoint.pt"
    echo ""
    print_status "To evaluate the regularized model:"
    print_status "  sbatch scripts/evaluate_model.sh \\"
    print_status "    --checkpoint-dir $OUTPUT_DIR \\"
    print_status "    --test-sets 'dev-clean test-clean'"
else
    print_error "Training failed with exit code $exit_code"
    print_status "Check the log file for errors: lact_asr_regularized.log"
    exit 1
fi
