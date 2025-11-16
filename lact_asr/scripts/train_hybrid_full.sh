#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -J lact_hybrid_full
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH -o lact_hybrid_full.log
#SBATCH --constraint=h200

# Training script for Hybrid LaCT + Wav2Vec2 ASR on FULL LibriSpeech (≈960h)

# Load HPC modules
ml load gcc/12.2
ml load cuda/12.2

# Set cache directories
export HUGGING_FACE_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_DATASETS_CACHE=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache
export HF_HOME=/nfs/stak/users/limjar/hpc-share/LaCT/lact_asr/.cache

# Activate virtual environment
source /nfs/stak/users/limjar/hpc-share/myVenv/bin/activate

# Weights & Biases (set your API key here or before sbatch)
# export WANDB_API_KEY="YOUR_API_KEY"
export WANDB_PROJECT=${WANDB_PROJECT:-lact-asr}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-hybrid_full_${SLURM_JOB_ID:-local}}
export WANDB_DIR=${WANDB_DIR:-./wandb}

# Set environment variables
export HOME="/nfs/stak/users/limjar"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /nfs/stak/users/limjar/hpc-share/LaCT/lact_asr

# Configuration
DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech"
OUTPUT_DIR="./checkpoints/hybrid_full"
CONFIG_PATH="./configs/hybrid_asr_config.json"
TRAIN_SUBSET="train-960"  # train-clean-100 + train-clean-360 + train-other-500
VAL_SUBSET="dev-clean"
TEST_SUBSET="test-clean"

# Hybrid-specific options
W2V2_MODEL="facebook/wav2vec2-base-960h"
FREEZE_ENCODER=0            # set to 1 to freeze wav2vec2 encoder
ENCODER_LR_SCALE="0.1"      # encoder uses 0.1x peak LR
ENABLE_TTT=1                # set to 0 to disable test-time training
TTT_LOSS_TYPE="masked_prediction"  # masked_prediction | entropy

# Training hyperparameters
BATCH_SIZE=8                # Hybrid model + encoder; keep memory moderate
MAX_EPOCHS=25
PEAK_LR="5e-4"
WARMUP_STEPS=10000
MAX_GRAD_NORM=0.5
GRADIENT_ACCUMULATION=2     # Effective batch size = 16
MAX_AUDIO_DURATION=25.0
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

verify_librispeech_subsets() {
    local root="$1"
    local subset="$2"
    if [[ "$subset" == "train-960" || "$subset" == "train-all" || "$subset" == "train-full" ]]; then
        local req=("train-clean-100" "train-clean-360" "train-other-500")
        for s in "${req[@]}"; do
            if [[ ! -d "$root/$s" ]]; then
                print_error "Required subset not found: $root/$s"
                return 1
            fi
        done
        return 0
    fi
    if [[ "$subset" == *","* ]]; then
        IFS=',' read -ra PARTS <<< "$subset"
        for s in "${PARTS[@]}"; do
            s_trimmed="$(echo "$s" | xargs)"
            if [[ ! -d "$root/$s_trimmed" ]]; then
                print_error "Subset not found: $root/$s_trimmed"
                return 1
            fi
        done
        return 0
    fi
    # Single subset
    [[ -d "$root/$subset" ]]
}

date
echo "🚀 Starting Hybrid LaCT + Wav2Vec2 ASR Training on Full LibriSpeech"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "Hybrid LaCT + Wav2Vec2 Full Training"
print_status "===================================="
print_status "Configuration:"
print_status "  Model config: $CONFIG_PATH"
print_status "  Wav2Vec2: $W2V2_MODEL (freeze=${FREEZE_ENCODER})"
print_status "  Encoder LR scale: $ENCODER_LR_SCALE"
print_status "  Use TTT: ${ENABLE_TTT} (loss=$TTT_LOSS_TYPE)"
print_status "  Data directory: $DATA_DIR"
print_status "  Training subset: $TRAIN_SUBSET (≈960h)"
print_status "  Validation subset: $VAL_SUBSET"
print_status "  Test subset (monitoring): $TEST_SUBSET"
print_status "  Output directory: $OUTPUT_DIR"
print_status ""
print_status "Training hyperparameters:"
print_status "  Batch size: $BATCH_SIZE (x${GRADIENT_ACCUMULATION} accumulation = effective $((BATCH_SIZE * GRADIENT_ACCUMULATION)))"
print_status "  Peak LR: $PEAK_LR (warmup $WARMUP_STEPS steps)"
print_status "  Max epochs: $MAX_EPOCHS"
print_status "  Gradient clipping: $MAX_GRAD_NORM"
print_status "  Mixed precision: DISABLED"
print_status "  Max audio duration: $MAX_AUDIO_DURATION seconds"
echo ""

# Preemption-safe checkpoint handling
SAFE_EXIT_SAVE_DIR="$OUTPUT_DIR/preempt_saves"
mkdir -p "$SAFE_EXIT_SAVE_DIR"
on_preempt() {
    echo ""
    print_warning "Received termination signal. Attempting to preserve latest checkpoint..."
    if [[ -f "$OUTPUT_DIR/latest_checkpoint.pt" ]]; then
        TS="$(date +%s)"
        cp -f "$OUTPUT_DIR/latest_checkpoint.pt" "$SAFE_EXIT_SAVE_DIR/latest_checkpoint_${TS}.pt"
        cp -f "$OUTPUT_DIR/latest_checkpoint.pt" "$OUTPUT_DIR/latest_checkpoint_preempt.pt"
        print_success "Saved preemption checkpoint: $SAFE_EXIT_SAVE_DIR/latest_checkpoint_${TS}.pt"
        print_success "Alias: $OUTPUT_DIR/latest_checkpoint_preempt.pt"
    else
        print_warning "No latest_checkpoint.pt found to copy."
    fi
    # Give Python time to flush logs
    sleep 2
    exit 0
}
trap on_preempt SIGTERM SIGINT

# Verify datasets
if ! verify_librispeech_subsets "$DATA_DIR" "$TRAIN_SUBSET"; then
    print_error "Training subset(s) missing under $DATA_DIR"
    print_status "Available subsets:"
    ls -1 "$DATA_DIR" 2>/dev/null || echo "  (cannot list directory)"
    echo ""
    print_status "Examples:"
    print_status "  TRAIN_SUBSET=train-960     # clean-100 + clean-360 + other-500"
    print_status "  TRAIN_SUBSET=\"train-clean-360,train-other-500\""
    exit 1
fi
print_success "Dataset verified for TRAIN_SUBSET=$TRAIN_SUBSET"
echo ""

# Check config exists
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

TRAIN_CMD="python training/train_hybrid_asr.py"
TRAIN_CMD="$TRAIN_CMD --config_path $CONFIG_PATH"
TRAIN_CMD="$TRAIN_CMD --data_dir $DATA_DIR"
TRAIN_CMD="$TRAIN_CMD --dataset_type librispeech"
TRAIN_CMD="$TRAIN_CMD --train_subset $TRAIN_SUBSET"
TRAIN_CMD="$TRAIN_CMD --val_subset $VAL_SUBSET"
TRAIN_CMD="$TRAIN_CMD --test_subset $TEST_SUBSET"
TRAIN_CMD="$TRAIN_CMD --output_dir $OUTPUT_DIR"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --max_epochs $MAX_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --peak_lr $PEAK_LR"
TRAIN_CMD="$TRAIN_CMD --warmup_steps $WARMUP_STEPS"
TRAIN_CMD="$TRAIN_CMD --encoder_lr_scale $ENCODER_LR_SCALE"
TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION"
TRAIN_CMD="$TRAIN_CMD --max_grad_norm $MAX_GRAD_NORM"
TRAIN_CMD="$TRAIN_CMD --max_audio_duration $MAX_AUDIO_DURATION"
TRAIN_CMD="$TRAIN_CMD --num_workers $NUM_WORKERS"
TRAIN_CMD="$TRAIN_CMD --device cuda"
TRAIN_CMD="$TRAIN_CMD --logging_steps 100"
TRAIN_CMD="$TRAIN_CMD --save_steps 5000"
TRAIN_CMD="$TRAIN_CMD --eval_steps 1000"
TRAIN_CMD="$TRAIN_CMD --wav2vec2_model_name $W2V2_MODEL"
if [[ "$FREEZE_ENCODER" -eq 1 ]]; then
    TRAIN_CMD="$TRAIN_CMD --freeze_encoder"
fi
if [[ "$ENABLE_TTT" -eq 1 ]]; then
    TRAIN_CMD="$TRAIN_CMD --enable_ttt --ttt_loss_type $TTT_LOSS_TYPE"
fi
if command -v wandb >/dev/null 2>&1 || python -c "import importlib,sys; sys.exit(0) if importlib.util.find_spec('wandb') else sys.exit(1)"; then
    # Enable WandB if available; you can also export WANDB_API_KEY before submitting
    TRAIN_CMD="$TRAIN_CMD --wandb --wandb_project lact-asr --wandb_run_name hybrid_full_${SLURM_JOB_ID:-local}"
    print_status "Weights & Biases logging ENABLED (project=lact-asr, run=${SLURM_JOB_ID:-local})"
else
    print_warning "wandb not installed; skipping online logging"
fi
TRAIN_CMD="$TRAIN_CMD $RESUME_FLAG"
# Note: NO --mixed_precision flag (disabled by default)

echo "=================================================="
print_status "Training Command:"
echo "$TRAIN_CMD"
echo "=================================================="
echo ""

print_status "Estimated training time: multi-day for full convergence on 960h"
print_status "Checkpoints saved every 5000 steps to: $OUTPUT_DIR"
print_status "Monitor progress: tail -f $PWD/lact_hybrid_full.log"
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
    print_status "To evaluate the model with KenLM beam search:"
    print_status "  python scripts/evaluate_asr_model.py --model-type lact \\"
    print_status "    --model-path $OUTPUT_DIR/best_model.pt --subset test-clean \\"
    print_status "    --lm-weight 3.23 --word-score -0.26 --beam-threshold 25"
else
    print_error "Training failed with exit code $exit_code"
    print_status "Check the log file for errors: lact_hybrid_full.log"
    exit 1
fi


