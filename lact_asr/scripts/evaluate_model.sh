#!/bin/bash
#SBATCH -t 0-04:00:00
#SBATCH -J lact_asr_evaluation
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o lact_asr_evaluation.log
#SBATCH --constraint=h200

# Comprehensive evaluation script for LaCT ASR model
# Computes WER, CER on test sets and generates detailed analysis

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
DEFAULT_OUTPUT_DIR="./evaluation_results"

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

list_checkpoints() {
    local checkpoint_dir="$1"
    if [[ ! -d "$checkpoint_dir" ]]; then
        print_error "Directory not found: $checkpoint_dir"
        return 1
    fi
    
    print_status "Available checkpoints in $checkpoint_dir:"
    echo ""
    
    # Check for best model
    if [[ -f "$checkpoint_dir/best_model.pt" ]]; then
        local size=$(du -h "$checkpoint_dir/best_model.pt" | cut -f1)
        local date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$checkpoint_dir/best_model.pt" 2>/dev/null || stat -c "%y" "$checkpoint_dir/best_model.pt" 2>/dev/null | cut -d' ' -f1,2)
        echo "  ✓ best_model.pt (${size}, ${date})"
    fi
    
    # Check for latest checkpoint
    if [[ -f "$checkpoint_dir/latest_checkpoint.pt" ]]; then
        local size=$(du -h "$checkpoint_dir/latest_checkpoint.pt" | cut -f1)
        local date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$checkpoint_dir/latest_checkpoint.pt" 2>/dev/null || stat -c "%y" "$checkpoint_dir/latest_checkpoint.pt" 2>/dev/null | cut -d' ' -f1,2)
        echo "  ✓ latest_checkpoint.pt (${size}, ${date})"
    fi
    
    # List step checkpoints
    local step_files=($(ls -t "$checkpoint_dir"/checkpoint-step-*.pt 2>/dev/null))
    if [[ ${#step_files[@]} -gt 0 ]]; then
        echo ""
        echo "  Step checkpoints (most recent first):"
        for file in "${step_files[@]}"; do
            local basename=$(basename "$file")
            local size=$(du -h "$file" | cut -f1)
            local date=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$file" 2>/dev/null || stat -c "%y" "$file" 2>/dev/null | cut -d' ' -f1,2)
            echo "    - $basename (${size}, ${date})"
        done
    fi
    echo ""
}

show_usage() {
    cat << EOF
Usage: $0 [options]

Comprehensive evaluation of trained LaCT ASR model.
Computes WER, CER, and other metrics on test sets.

Options:
  -c, --checkpoint-dir DIR    Directory containing trained model (default: $DEFAULT_CHECKPOINT_DIR)
  -m, --checkpoint-file FILE  Specific checkpoint to evaluate. Can be:
                              - Just filename: checkpoint-step-5000.pt (uses -c dir)
                              - Relative path: ./checkpoints/my_model/checkpoint-step-5000.pt
                              - Absolute path: /full/path/to/checkpoint-step-5000.pt
                              If not specified, uses best_model.pt or latest_checkpoint.pt
  -d, --data-dir DIR          Directory with LibriSpeech dataset (default: $DEFAULT_DATA_DIR)
  -o, --output-dir DIR        Directory for evaluation results (default: $DEFAULT_OUTPUT_DIR)
  --test-sets SETS            Space-separated test sets (default: "dev-clean test-clean")
  --beam-width WIDTH          Beam width for decoding (default: 1)
  --lm-weight FLOAT           Language model weight (default: 3.23)
  --word-score FLOAT          Word insertion score (default: -0.26)
  --beam-threshold INT        Beam pruning threshold (default: 25)
  --max-samples NUM           Max samples per test set (default: all)
  --list-checkpoints          List available checkpoints in checkpoint directory and exit
  -h, --help                  Show this help message

Examples:
  $0                                              # Evaluate best_model.pt from default dir
  $0 -c ./checkpoints/my_model                   # Evaluate best from specific dir
  $0 -m checkpoint-step-5000.pt                  # Evaluate specific file (uses default dir)
  $0 -c ./checkpoints/my_model -m checkpoint-step-10000.pt  # Specific dir + file
  $0 -m ./checkpoints/regularized/checkpoint-step-15000.pt  # Full relative path
  $0 -m /nfs/stak/.../checkpoint-step-20000.pt   # Full absolute path
  $0 --test-sets "dev-clean"                     # Evaluate only on dev-clean
  $0 --beam-width 5                              # Use beam search decoding
  $0 --lm-weight 2.5 --word-score -0.5           # Tune decoder hyperparameters
  $0 --max-samples 500                           # Evaluate on first 500 samples
  $0 --list-checkpoints -c ./checkpoints/regularized  # List checkpoints

EOF
}

# Parse arguments
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
CHECKPOINT_FILE=""
DATA_DIR="$DEFAULT_DATA_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
TEST_SETS="dev-clean test-clean"
BEAM_WIDTH=1
LM_WEIGHT=5
WORD_SCORE=-0.26
BEAM_THRESHOLD=25
MAX_SAMPLES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -m|--checkpoint-file)
            CHECKPOINT_FILE="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --test-sets)
            TEST_SETS="$2"
            shift 2
            ;;
        --beam-width)
            BEAM_WIDTH="$2"
            shift 2
            ;;
        --lm-weight)
            LM_WEIGHT="$2"
            shift 2
            ;;
        --word-score)
            WORD_SCORE="$2"
            shift 2
            ;;
        --beam-threshold)
            BEAM_THRESHOLD="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --list-checkpoints)
            list_checkpoints "$CHECKPOINT_DIR"
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Allow test set names as positional arguments
            # Check if this is a valid test set name
            case "$1" in
                dev-clean|test-clean|train-clean-100|train-clean-360|train-other-500)
                    # Add to test sets if not already present
                    if [[ "$TEST_SETS" != *"$1"* ]]; then
                        TEST_SETS="$TEST_SETS $1"
                    fi
                    ;;
                *)
                    print_error "Unknown test set: $1"
                    print_error "Valid test sets: dev-clean, test-clean, train-clean-100, train-clean-360, train-other-500"
                    exit 1
                    ;;
            esac
            shift
            ;;
    esac
done

date
echo "📊 Starting LaCT ASR Comprehensive Evaluation"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR Model Evaluation"
print_status "========================="
print_status "Checkpoint directory: $CHECKPOINT_DIR"
if [[ -n "$CHECKPOINT_FILE" ]]; then
    print_status "Checkpoint file: $CHECKPOINT_FILE"
fi
print_status "Data directory: $DATA_DIR"
print_status "Output directory: $OUTPUT_DIR"
print_status "Test sets: $TEST_SETS"
print_status "Decoder parameters:"
print_status "  Beam width: $BEAM_WIDTH"
print_status "  LM weight: $LM_WEIGHT"
print_status "  Word score: $WORD_SCORE"
print_status "  Beam threshold: $BEAM_THRESHOLD"
print_status "Max samples per set: ${MAX_SAMPLES:-all}"
echo ""

# Determine which checkpoint file to use
if [[ -n "$CHECKPOINT_FILE" ]]; then
    # User specified a checkpoint file
    # Check if it's an absolute path or contains directory separators
    if [[ "$CHECKPOINT_FILE" == /* ]] || [[ "$CHECKPOINT_FILE" == */* ]]; then
        # It's a full/relative path, use it as-is
        MODEL_FILE="$CHECKPOINT_FILE"
        # Update CHECKPOINT_DIR to the directory of this file
        CHECKPOINT_DIR="$(dirname "$MODEL_FILE")"
        CHECKPOINT_FILE="$(basename "$MODEL_FILE")"
    else
        # It's just a filename, combine with checkpoint dir
        MODEL_FILE="$CHECKPOINT_DIR/$CHECKPOINT_FILE"
    fi
    
    # Check if checkpoint directory exists
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        print_error "Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
    
    if [[ ! -f "$MODEL_FILE" ]]; then
        print_error "Specified checkpoint file not found: $MODEL_FILE"
        print_status "Available checkpoints in $CHECKPOINT_DIR:"
        ls -lh "$CHECKPOINT_DIR"/*.pt 2>/dev/null || echo "  No .pt files found"
        exit 1
    fi
    print_success "Using specified checkpoint: $CHECKPOINT_FILE"
else
    # Check if checkpoint directory exists (for auto-select mode)
    if [[ ! -d "$CHECKPOINT_DIR" ]]; then
        print_error "Checkpoint directory not found: $CHECKPOINT_DIR"
        exit 1
    fi
    
    # Auto-select: best_model.pt > latest_checkpoint.pt > any checkpoint-step-*.pt
    MODEL_FILE="$CHECKPOINT_DIR/best_model.pt"
    if [[ ! -f "$MODEL_FILE" ]]; then
        MODEL_FILE="$CHECKPOINT_DIR/latest_checkpoint.pt"
        if [[ ! -f "$MODEL_FILE" ]]; then
            # Try to find any checkpoint-step-*.pt file
            STEP_CHECKPOINTS=($(ls -t "$CHECKPOINT_DIR"/checkpoint-step-*.pt 2>/dev/null))
            if [[ ${#STEP_CHECKPOINTS[@]} -gt 0 ]]; then
                MODEL_FILE="${STEP_CHECKPOINTS[0]}"
                print_warning "Using most recent step checkpoint: $(basename $MODEL_FILE)"
            else
                print_error "No model checkpoint found in $CHECKPOINT_DIR"
                print_status "Expected files: best_model.pt, latest_checkpoint.pt, or checkpoint-step-*.pt"
                exit 1
            fi
        else
            print_warning "Using latest_checkpoint.pt (best_model.pt not found)"
        fi
    else
        print_success "Using best_model.pt"
    fi
fi

CONFIG_FILE="$CHECKPOINT_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_success "Using checkpoint: $(basename $MODEL_FILE)"
print_success "Full path: $MODEL_FILE"
print_success "Config: $CONFIG_FILE"

# Show checkpoint info if available
if [[ -f "$MODEL_FILE" ]]; then
    CHECKPOINT_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    print_status "Checkpoint size: $CHECKPOINT_SIZE"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comprehensive evaluation
print_status "Running comprehensive evaluation..."
echo ""
echo "=================================================="

# Build the Python command with arguments
PYTHON_CMD="python scripts/evaluate_asr_model.py \
    --model-path \"$MODEL_FILE\" \
    --config-path \"$CONFIG_FILE\" \
    --data-dir \"$DATA_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --test-sets $TEST_SETS \
    --beam-width $BEAM_WIDTH \
    --lm-weight $LM_WEIGHT \
    --word-score $WORD_SCORE \
    --beam-threshold $BEAM_THRESHOLD \
    --batch-size 8 \
    --device cuda \
    --move-emission-to-cpu"

# Add max-samples if specified
if [[ -n "$MAX_SAMPLES" ]]; then
    PYTHON_CMD="$PYTHON_CMD --max-samples $MAX_SAMPLES"
fi

# Execute the Python evaluation script
eval $PYTHON_CMD

exit_code=$?

echo ""
echo "=================================================="
echo ""
date

if [[ $exit_code -eq 0 ]]; then
    print_success "Evaluation completed successfully!"
    print_status "Results saved to: $OUTPUT_DIR"
    print_status "  - evaluation_results.json (detailed metrics)"
    print_status "  - evaluation_report.md (formatted report)"
    echo ""
    print_status "View results:"
    print_status "  cat $OUTPUT_DIR/evaluation_report.md"
    print_status "  cat $OUTPUT_DIR/evaluation_results.json"
else
    print_error "Evaluation failed with exit code $exit_code"
    exit 1
fi

