#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH -J lact_model_comparison
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o lact_model_comparison.log
#SBATCH --constraint=h200

# Compare the overfitted large model vs the new regularized model

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
echo "📊 LaCT ASR Model Comparison"
echo "Working directory: $(pwd)"
echo ""

print_status "Comparing Model Performance"
print_status "============================"
echo ""

# Model 1: Large overfitted model
MODEL1_DIR="./checkpoints/librispeech_improved_full"
MODEL1_NAME="Large Model (1024d, 18L)"

# Model 2: Regularized model (will be created)
MODEL2_DIR="./checkpoints/librispeech_regularized"
MODEL2_NAME="Regularized Model (640d, 12L)"

print_status "Model 1: $MODEL1_NAME"
print_status "  Directory: $MODEL1_DIR"
print_status "  Architecture: 1024 hidden, 18 layers, 8 LaCT heads"
print_status "  Parameters: ~311M"
print_status "  Status: Overfitted (training loss 0.089, validation loss 0.26)"
echo ""

print_status "Model 2: $MODEL2_NAME"
print_status "  Directory: $MODEL2_DIR"
print_status "  Architecture: 640 hidden, 12 layers, 4 LaCT heads"
print_status "  Parameters: ~150M"
print_status "  Status: Not yet trained (needs to be created)"
echo ""

# Check if Model 1 exists
if [[ -f "$MODEL1_DIR/best_model.pt" ]]; then
    print_success "✓ Model 1 found: $MODEL1_DIR/best_model.pt"
else
    print_error "✗ Model 1 not found: $MODEL1_DIR/best_model.pt"
    exit 1
fi

# Check if Model 2 exists
if [[ -f "$MODEL2_DIR/best_model.pt" ]]; then
    print_success "✓ Model 2 found: $MODEL2_DIR/best_model.pt"
    MODEL2_READY=true
else
    print_warning "⚠ Model 2 not yet trained: $MODEL2_DIR/best_model.pt"
    print_status "Train it first with: sbatch scripts/train_regularized.sh"
    MODEL2_READY=false
fi

echo ""

if [[ "$MODEL2_READY" == "true" ]]; then
    print_status "Running evaluation on both models..."
    echo ""
    
    # Evaluate Model 1
    print_status "Evaluating Model 1 (Large Overfitted Model)..."
    echo "=================================================="
    sbatch scripts/evaluate_model.sh \
        --checkpoint-dir "$MODEL1_DIR" \
        --test-sets "dev-clean test-clean" \
        --output-dir "./evaluation_results/large_model"
    
    # Evaluate Model 2
    print_status "Evaluating Model 2 (Regularized Model)..."
    echo "=================================================="
    sbatch scripts/evaluate_model.sh \
        --checkpoint-dir "$MODEL2_DIR" \
        --test-sets "dev-clean test-clean" \
        --output-dir "./evaluation_results/regularized_model"
    
    echo ""
    print_success "Both evaluations submitted!"
    print_status "Results will be saved to:"
    print_status "  - ./evaluation_results/large_model/"
    print_status "  - ./evaluation_results/regularized_model/"
    print_status ""
    print_status "Expected differences:"
    print_status "  ✓ Large model: High WER on test-clean (overfitting)"
    print_status "  ✓ Regularized model: Lower WER on test-clean (better generalization)"
    
else
    print_status "To compare models:"
    print_status "1. Train the regularized model:"
    print_status "   sbatch scripts/train_regularized.sh"
    print_status ""
    print_status "2. Then run this comparison:"
    print_status "   sbatch scripts/compare_models.sh"
    print_status ""
    print_status "3. Compare results in:"
    print_status "   - ./evaluation_results/large_model/"
    print_status "   - ./evaluation_results/regularized_model/"
fi

echo ""
date
