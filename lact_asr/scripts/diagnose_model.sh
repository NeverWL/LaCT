#!/bin/bash
#SBATCH -t 0-00:30:00
#SBATCH -J lact_asr_diagnose
#SBATCH -A eecs
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o lact_asr_diagnose.log

# Diagnostic script for LaCT ASR model initialization
# This script checks if the model can initialize and perform forward passes without NaN

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
echo "🔍 Starting LaCT ASR Model Diagnostic"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"
nvidia-smi
echo ""

print_status "LaCT ASR Model Diagnostic"
print_status "=========================="
print_status "Testing model initialization and forward pass"
print_status "This will help identify NaN issues"
echo ""

# Check Python environment
print_status "Checking Python environment..."
python --version
print_status "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
print_status "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
print_status "CUDA version: $(python -c 'import torch; print(torch.version.cuda if torch.cuda.is_available() else "N/A")')"
echo ""

# Check required packages
print_status "Checking required packages..."
python_deps=("torch" "torchaudio" "transformers" "fla")
missing_deps=()

for dep in "${python_deps[@]}"; do
    if ! python -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
    fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    print_error "Missing Python dependencies: ${missing_deps[*]}"
    print_status "Please install missing dependencies first"
    exit 1
fi

print_success "All required packages available"
echo ""

# Configuration
DATA_DIR="/nfs/stak/users/limjar/hpc-share/datasets/LibriSpeech_LaCT/LibriSpeech"
SUBSET="train-clean-100"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -s|--subset)
            SUBSET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Data directory: $DATA_DIR"
print_status "Subset: $SUBSET"
echo ""

# Run diagnostic script
print_status "Running model diagnostic..."
echo ""
echo "=================================================="

# Run with dummy data first, then real data
if [[ -d "$DATA_DIR" ]]; then
    print_status "Testing with both dummy and real data..."
    python scripts/diagnose_model.py --data-dir "$DATA_DIR" --subset "$SUBSET"
else
    print_warning "Data directory not found: $DATA_DIR"
    print_status "Testing with dummy data only..."
    python scripts/diagnose_model.py
fi

exit_code=$?

echo "=================================================="
echo ""
date

if [[ $exit_code -eq 0 ]]; then
    print_success "Model diagnostic completed successfully!"
    print_status "Model appears to be healthy and can perform forward passes"
    print_status "The NaN issue may be related to:"
    print_status "  - Specific data samples causing issues"
    print_status "  - Training dynamics (gradient accumulation, learning rate schedule)"
    print_status "  - Interaction between mixed precision and model architecture"
    echo ""
    print_status "Next steps:"
    print_status "  1. Check training logs for which specific batches cause NaN"
    print_status "  2. Try training with smaller batch size (--batch_size 2)"
    print_status "  3. Try without mixed precision"
    print_status "  4. Reduce learning rate further (--learning_rate 1e-6)"
else
    print_error "Model diagnostic FAILED!"
    print_error "The model produces NaN during initialization or forward pass"
    print_status "This indicates a fundamental issue with:"
    print_status "  - Model initialization (weights initialized to NaN/Inf)"
    print_status "  - Architecture configuration (incompatible dimensions)"
    print_status "  - TTT/LaCT layer implementation"
    echo ""
    print_status "Recommended actions:"
    print_status "  1. Check the diagnostic output above for which parameter has NaN"
    print_status "  2. Try a smaller model configuration"
    print_status "  3. Check if fla (flash-linear-attention) is properly installed"
    print_status "  4. Review model initialization in modeling_lact_asr.py"
    exit 1
fi
