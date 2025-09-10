#!/bin/bash
#SBATCH -t 0-24:00:00
#SBATCH -J lact_asr_librispeech
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:2
#SBATCH --mem=240G
#SBATCH -o lact_asr_librispeech.log
#SBATCH --constraint=h200

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
cd /nfs/stak/users/limjar/hpc-share/LaCT/LaCT/lact_asr

# Default parameters
LIBRISPEECH_ROOT="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech"
OUTPUT_DIR="./checkpoints/librispeech_base"
CONFIG_PATH="./configs/base_asr_config.json"
BATCH_SIZE="8"
MAX_EPOCHS="20"
LEARNING_RATE="5e-4"
GRADIENT_ACCUMULATION_STEPS="2"
MAX_AUDIO_DURATION="20.0"
NUM_WORKERS="4"
DEVICE="cuda"
TRAIN_SUBSET="train-clean-100"
VAL_SUBSET="dev-clean"

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -d, --data-dir PATH         LibriSpeech dataset directory"
    echo "  -o, --output-dir PATH       Output directory for checkpoints"
    echo "  -c, --config PATH           Model configuration file"
    echo "  -b, --batch-size SIZE       Batch size (default: 8)"
    echo "  -e, --epochs NUM            Maximum epochs (default: 20)"
    echo "  -l, --learning-rate RATE    Learning rate (default: 5e-4)"
    echo "  -g, --grad-accum STEPS      Gradient accumulation steps (default: 2)"
    echo "  --max-duration SEC          Maximum audio duration (default: 20.0)"
    echo "  --train-subset NAME         Training subset (default: train-clean-100)"
    echo "  --val-subset NAME           Validation subset (default: dev-clean)"
    echo "  --num-workers NUM           Number of data workers (default: 4)"
    echo "  --device DEVICE             Device to use (default: cuda)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Train with default settings"
    echo "  $0 -b 16 -e 30                      # Larger batch size, more epochs"
    echo "  $0 --train-subset train-clean-360   # Use larger training set"
    echo "  $0 -d /path/to/LibriSpeech          # Custom dataset path"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            LIBRISPEECH_ROOT="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
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
        -g|--grad-accum)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --max-duration)
            MAX_AUDIO_DURATION="$2"
            shift 2
            ;;
        --train-subset)
            TRAIN_SUBSET="$2"
            shift 2
            ;;
        --val-subset)
            VAL_SUBSET="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
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

date
echo "🚀 Starting LaCT ASR training on LibriSpeech"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

# Check if LibriSpeech directory exists
if [ ! -d "$LIBRISPEECH_ROOT" ]; then
    echo "❌ Error: LibriSpeech directory not found at $LIBRISPEECH_ROOT"
    echo "Please download LibriSpeech and provide the correct path"
    echo "Usage: $0 --data-dir /path/to/LibriSpeech"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check for existing checkpoints and resume if available
RESUME_CHECKPOINT=""
if [ -f "$OUTPUT_DIR/latest_checkpoint.pt" ]; then
    echo "🔄 Found existing checkpoint, will resume training..."
    RESUME_CHECKPOINT="--resume_from_checkpoint \"$OUTPUT_DIR/latest_checkpoint.pt\""
else
    echo "🆕 Starting fresh training (no existing checkpoint found)"
fi

# Build training command
TRAIN_CMD="python training/train_asr.py"
TRAIN_CMD="$TRAIN_CMD --config_path \"$CONFIG_PATH\""
TRAIN_CMD="$TRAIN_CMD --dataset_type librispeech"
TRAIN_CMD="$TRAIN_CMD --data_dir \"$LIBRISPEECH_ROOT\""
TRAIN_CMD="$TRAIN_CMD --train_subset \"$TRAIN_SUBSET\""
TRAIN_CMD="$TRAIN_CMD --val_subset \"$VAL_SUBSET\""
TRAIN_CMD="$TRAIN_CMD --output_dir \"$OUTPUT_DIR\""
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --max_epochs $MAX_EPOCHS"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
TRAIN_CMD="$TRAIN_CMD --max_audio_duration $MAX_AUDIO_DURATION"
TRAIN_CMD="$TRAIN_CMD --mixed_precision"
TRAIN_CMD="$TRAIN_CMD --num_workers $NUM_WORKERS"
TRAIN_CMD="$TRAIN_CMD --device \"$DEVICE\""
TRAIN_CMD="$TRAIN_CMD --logging_steps 50"
TRAIN_CMD="$TRAIN_CMD --save_steps 1000"
TRAIN_CMD="$TRAIN_CMD --eval_steps 500"
TRAIN_CMD="$TRAIN_CMD $RESUME_CHECKPOINT"

echo "🎯 Training configuration:"
echo "  LibriSpeech root: $LIBRISPEECH_ROOT"
echo "  Output directory: $OUTPUT_DIR"
echo "  Config file: $CONFIG_PATH"
echo "  Training subset: $TRAIN_SUBSET"
echo "  Validation subset: $VAL_SUBSET"
echo "  Batch size: $BATCH_SIZE"
echo "  Max epochs: $MAX_EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  Max audio duration: $MAX_AUDIO_DURATION"
echo "  Number of workers: $NUM_WORKERS"
echo "  Device: $DEVICE"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "  Resume from: $OUTPUT_DIR/latest_checkpoint.pt"
fi
echo ""
echo "Command: $TRAIN_CMD"
echo ""

# Execute training
eval $TRAIN_CMD

echo ""
date
echo "✅ LaCT ASR training completed! Model saved to $OUTPUT_DIR"
