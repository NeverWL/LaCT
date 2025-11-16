#!/bin/bash
#
# LibriSpeech Dataset Download Script for LaCT ASR
# This script downloads LibriSpeech dataset using HuggingFace datasets
# (Works in HPC environments where openslr.org may be blocked)
#

set -e  # Exit on any error

# Default configuration
DEFAULT_DATA_DIR="/tmp/LibriSpeech"
DEFAULT_SUBSETS="train-clean-100 dev-clean"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [options]

Downloads LibriSpeech dataset for LaCT ASR training using HuggingFace datasets.
This method works in HPC environments where direct HTTP access may be blocked.

Options:
  -d, --data-dir DIR          Directory to download LibriSpeech (default: $DEFAULT_DATA_DIR)
  -s, --subsets LIST          Space-separated list of subsets to download (default: "$DEFAULT_SUBSETS")
  -f, --force                 Force re-download even if files exist
  --train-only               Download only training data (train-clean-100)
  --full                     Download full dataset (train-clean-100, train-clean-360, dev-clean, test-clean)
  --minimal                  Download minimal dataset (train-clean-100, dev-clean) - same as default
  --no-validate              Skip validation after download
  -h, --help                 Show this help message

Available subsets:
  train-clean-100    100 hours of clean training speech
  train-clean-360    360 hours of clean training speech  
  train-other-500    500 hours of other training speech
  dev-clean          Development set (clean)
  dev-other          Development set (other)
  test-clean         Test set (clean)
  test-other         Test set (other)

Examples:
  $0                                           # Download minimal dataset to default location
  $0 -d /data/LibriSpeech                    # Download to custom directory
  $0 --full                                   # Download full dataset
  $0 -s "train-clean-360 dev-clean"          # Download specific subsets
  $0 --train-only                            # Download only training data

After download, use with training script:
  ./examples/train_librispeech.sh --data-dir /path/to/LibriSpeech

EOF
}

# Function to check if a subset is valid
is_valid_subset() {
    local subset=$1
    local valid_subsets="train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other"
    
    for valid in $valid_subsets; do
        if [[ "$subset" == "$valid" ]]; then
            return 0
        fi
    done
    return 1
}

# Parse command line arguments
DATA_DIR="$DEFAULT_DATA_DIR"
SUBSETS="$DEFAULT_SUBSETS"
FORCE=false
NO_VALIDATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -s|--subsets)
            SUBSETS="$2"
            shift 2
            ;;
        -k|--keep-archives)
            # Ignored for HF download (no archives)
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --train-only)
            SUBSETS="train-clean-100"
            shift
            ;;
        --full)
            SUBSETS="train-clean-100 train-clean-360 train-other-500 dev-clean test-clean"
            shift
            ;;
        --minimal)
            SUBSETS="train-clean-100 dev-clean"
            shift
            ;;
        --no-validate)
            NO_VALIDATE=true
            shift
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

# Validate subsets
for subset in $SUBSETS; do
    if ! is_valid_subset "$subset"; then
        print_error "Invalid subset: $subset"
        print_error "Run '$0 --help' to see available subsets"
        exit 1
    fi
done

# Check Python dependencies
print_status "Checking Python dependencies..."
missing_deps=()

for dep in datasets soundfile tqdm numpy; do
    if ! python -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
fi
done

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    print_error "Missing required Python packages: ${missing_deps[*]}"
    print_status "Installing missing dependencies..."
    pip install datasets soundfile tqdm numpy
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to install dependencies"
        exit 1
    fi
fi

print_success "All dependencies available"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/download_librispeech_hf.py"

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    print_error "Python download script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Print configuration
echo "=================================================="
echo "LibriSpeech Download Configuration"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Subsets to download: $SUBSETS"
echo "Force re-download: $FORCE"
echo "Skip validation: $NO_VALIDATE"
echo "=================================================="
echo ""

# Estimate download size
total_size_gb=0
for subset in $SUBSETS; do
    case $subset in
        train-clean-100) total_size_gb=$((total_size_gb + 6)) ;;
        train-clean-360) total_size_gb=$((total_size_gb + 23)) ;;
        train-other-500) total_size_gb=$((total_size_gb + 30)) ;;
        dev-clean) total_size_gb=$((total_size_gb + 1)) ;;
        dev-other) total_size_gb=$((total_size_gb + 1)) ;;
        test-clean) total_size_gb=$((total_size_gb + 1)) ;;
        test-other) total_size_gb=$((total_size_gb + 1)) ;;
    esac
done

print_status "Estimated download size: ~${total_size_gb} GB"
print_status "Make sure you have sufficient disk space"
print_status "Download method: HuggingFace datasets (works in restricted networks)"
echo ""

# SLURM non-interactive mode - proceed automatically
print_status "Proceeding with download automatically (SLURM mode)"

# Check if dataset already exists
LIBRISPEECH_PATH="$DATA_DIR/LibriSpeech"
if [[ -d "$LIBRISPEECH_PATH" && "$FORCE" != "true" ]]; then
    print_warning "Dataset already exists at $LIBRISPEECH_PATH"
    
    # Check which requested subsets are missing
    missing_subsets=()
    for subset in $SUBSETS; do
        if [[ ! -d "$LIBRISPEECH_PATH/$subset" ]]; then
            missing_subsets+=("$subset")
        fi
    done
    
    if [[ ${#missing_subsets[@]} -eq 0 ]]; then
        print_status "All requested subsets already exist"
        print_status "Using existing dataset (SLURM non-interactive mode)"
        exit 0
    else
        print_warning "Missing subsets: ${missing_subsets[*]}"
        SUBSETS="${missing_subsets[*]}"
        print_status "Will download only missing subsets: $SUBSETS"
    fi
fi

# Start download process
print_status "Starting LibriSpeech download via HuggingFace..."
start_time=$(date +%s)

# Build Python command
PYTHON_CMD="python $PYTHON_SCRIPT --data-dir $DATA_DIR --subsets \"$SUBSETS\""
if [[ "$NO_VALIDATE" == "true" ]]; then
    PYTHON_CMD="$PYTHON_CMD --no-validate"
fi

# Execute Python download script
eval $PYTHON_CMD

exit_code=$?

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

if [[ $exit_code -eq 0 ]]; then
echo ""
echo "=================================================="
print_success "LibriSpeech download completed successfully!"
echo "=================================================="
    print_success "Downloaded to: $LIBRISPEECH_PATH"
print_success "Time taken: ${hours}h ${minutes}m ${seconds}s"
print_success "Subsets downloaded: $SUBSETS"
echo ""
print_status "Next steps:"
    echo "  1. Start training:"
    echo "     cd $(dirname "$SCRIPT_DIR")"
    echo "     ./examples/train_librispeech.sh --data-dir $LIBRISPEECH_PATH"
echo ""
    echo "  2. Or use the setup script (will skip download):"
    echo "     ./scripts/setup_and_train.sh --skip-download --data-dir $(dirname $LIBRISPEECH_PATH)"
echo "=================================================="
else
    print_error "Download failed with exit code $exit_code"
    exit $exit_code
fi
