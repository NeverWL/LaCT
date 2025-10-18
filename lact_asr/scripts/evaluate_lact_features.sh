#!/bin/bash
#SBATCH -t 0-01:00:00
#SBATCH -J lact_features_eval
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o lact_features_evaluation.log
#SBATCH --constraint=h200

# Evaluate LaCT-specific features as described in the paper
# Focus on: GPU utilization, large chunks, linear scaling, TTT adaptation

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

# Parse arguments
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
DATA_DIR="$DEFAULT_DATA_DIR"
TEST_SUBSET="dev-clean"
OUTPUT_FILE="lact_features_evaluation.json"
NUM_BATCHES=20

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
        --test-subset)
            TEST_SUBSET="$2"
            shift 2
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --num-batches)
            NUM_BATCHES="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            # Allow test subset as positional argument
            TEST_SUBSET="$1"
            shift
            ;;
    esac
done

date
echo "🔬 LaCT Features Evaluation (Based on arXiv:2505.23884)"
echo "Working directory: $(pwd)"
nvidia-smi
echo ""

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Data: $DATA_DIR/$TEST_SUBSET"
echo "  Output: $OUTPUT_FILE"
echo ""

# Run evaluation
python scripts/evaluate_lact_features.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --data-dir "$DATA_DIR" \
    --test-subset "$TEST_SUBSET" \
    --output-file "$OUTPUT_FILE" \
    --num-batches "$NUM_BATCHES"

exit_code=$?

echo ""
date

if [[ $exit_code -eq 0 ]]; then
    echo "✅ LaCT features evaluation completed!"
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "This evaluation demonstrates:"
    echo "  1. High GPU utilization from large chunks (vs <5% for small-batch TTT)"
    echo "  2. Linear O(n) scaling (vs O(n²) for standard attention)"  
    echo "  3. Test-time adaptation with scalable fast weights"
    echo "  4. Efficient long-form audio processing"
else
    echo "❌ Evaluation failed"
    exit 1
fi

