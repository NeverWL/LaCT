#!/bin/bash
#
# LibriSpeech Dataset Download Script for LaCT ASR
# This script downloads LibriSpeech dataset in the format expected by the training pipeline
#

set -e  # Exit on any error

# Default configuration
DEFAULT_DATA_DIR="/tmp/LibriSpeech"
DEFAULT_SUBSETS="train-clean-100 dev-clean"
BASE_URL="http://www.openslr.org/resources/12"

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

Downloads LibriSpeech dataset for LaCT ASR training.

Options:
  -d, --data-dir DIR          Directory to download LibriSpeech (default: $DEFAULT_DATA_DIR)
  -s, --subsets LIST          Space-separated list of subsets to download (default: "$DEFAULT_SUBSETS")
  -k, --keep-archives         Keep downloaded tar.gz files after extraction
  -f, --force                 Force re-download even if files exist
  --train-only               Download only training data (train-clean-100)
  --full                     Download full dataset (train-clean-100, train-clean-360, dev-clean, test-clean)
  --minimal                  Download minimal dataset (train-clean-100, dev-clean) - same as default
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

# Function to get file size from URL
get_remote_file_size() {
    local url=$1
    curl -sI "$url" | grep -i content-length | awk '{print $2}' | tr -d '\r'
}

# Function to format bytes
format_bytes() {
    local bytes=$1
    if [[ $bytes -gt 1073741824 ]]; then
        echo "$(( bytes / 1073741824 )) GB"
    elif [[ $bytes -gt 1048576 ]]; then
        echo "$(( bytes / 1048576 )) MB"
    elif [[ $bytes -gt 1024 ]]; then
        echo "$(( bytes / 1024 )) KB"
    else
        echo "$bytes bytes"
    fi
}

# Function to download and extract a subset
download_subset() {
    local subset=$1
    local data_dir=$2
    local keep_archives=$3
    local force=$4
    
    local archive_name="${subset}.tar.gz"
    local archive_path="${data_dir}/${archive_name}"
    local extract_path="${data_dir}/LibriSpeech/${subset}"
    local download_url="${BASE_URL}/${archive_name}"
    
    print_status "Processing subset: $subset"
    
    # Check if already extracted and not forcing
    if [[ -d "$extract_path" && "$force" != "true" ]]; then
        print_warning "Subset $subset already exists at $extract_path (use --force to re-download)"
        return 0
    fi
    
    # Create data directory
    mkdir -p "$data_dir"
    
    # Check if archive exists and is complete
    local should_download=true
    if [[ -f "$archive_path" && "$force" != "true" ]]; then
        print_status "Archive $archive_name already exists, checking integrity..."
        
        # Get remote and local file sizes
        local remote_size=$(get_remote_file_size "$download_url")
        local local_size=$(stat -f%z "$archive_path" 2>/dev/null || stat -c%s "$archive_path" 2>/dev/null)
        
        if [[ -n "$remote_size" && "$local_size" -eq "$remote_size" ]]; then
            print_success "Archive $archive_name is complete ($(format_bytes $local_size))"
            should_download=false
        else
            print_warning "Archive $archive_name is incomplete or corrupted, will re-download"
            rm -f "$archive_path"
        fi
    fi
    
    # Download if needed
    if [[ "$should_download" == "true" ]]; then
        print_status "Downloading $subset from $download_url"
        
        # Get file size for progress
        local file_size=$(get_remote_file_size "$download_url")
        if [[ -n "$file_size" ]]; then
            print_status "File size: $(format_bytes $file_size)"
        fi
        
        # Download with progress bar
        if command -v wget >/dev/null 2>&1; then
            wget --progress=bar:force:noscroll -O "$archive_path" "$download_url"
        elif command -v curl >/dev/null 2>&1; then
            curl -L --progress-bar -o "$archive_path" "$download_url"
        else
            print_error "Neither wget nor curl found. Please install one of them."
            return 1
        fi
        
        if [[ $? -ne 0 ]]; then
            print_error "Failed to download $subset"
            rm -f "$archive_path"
            return 1
        fi
        
        print_success "Downloaded $archive_name"
    fi
    
    # Extract archive
    print_status "Extracting $archive_name..."
    
    # Remove existing extraction if forcing
    if [[ "$force" == "true" && -d "$extract_path" ]]; then
        rm -rf "$extract_path"
    fi
    
    # Extract with progress
    if command -v pv >/dev/null 2>&1; then
        pv "$archive_path" | tar -xzf - -C "$data_dir"
    else
        tar -xzf "$archive_path" -C "$data_dir"
    fi
    
    if [[ $? -ne 0 ]]; then
        print_error "Failed to extract $archive_name"
        return 1
    fi
    
    print_success "Extracted $subset to $extract_path"
    
    # Remove archive if not keeping
    if [[ "$keep_archives" != "true" ]]; then
        rm -f "$archive_path"
        print_status "Removed archive $archive_name"
    fi
    
    return 0
}

# Function to verify dataset structure
verify_dataset() {
    local data_dir=$1
    local subsets=$2
    
    print_status "Verifying dataset structure..."
    
    local librispeech_dir="${data_dir}/LibriSpeech"
    if [[ ! -d "$librispeech_dir" ]]; then
        print_error "LibriSpeech directory not found at $librispeech_dir"
        return 1
    fi
    
    local total_files=0
    local total_hours=0
    
    for subset in $subsets; do
        local subset_dir="${librispeech_dir}/${subset}"
        if [[ ! -d "$subset_dir" ]]; then
            print_warning "Subset directory not found: $subset_dir"
            continue
        fi
        
        # Count audio files
        local audio_count=$(find "$subset_dir" -name "*.flac" | wc -l | tr -d ' ')
        local transcript_count=$(find "$subset_dir" -name "*.trans.txt" | wc -l | tr -d ' ')
        
        print_success "Subset $subset: $audio_count audio files, $transcript_count transcript files"
        total_files=$((total_files + audio_count))
        
        # Estimate hours based on subset name
        case $subset in
            *100*) total_hours=$((total_hours + 100)) ;;
            *360*) total_hours=$((total_hours + 360)) ;;
            *500*) total_hours=$((total_hours + 500)) ;;
            *clean*|*other*) total_hours=$((total_hours + 5)) ;;  # Dev/test sets are ~5 hours each
        esac
    done
    
    print_success "Total: $total_files audio files, approximately $total_hours hours of speech"
    
    # Check for common issues
    local sample_audio=$(find "$librispeech_dir" -name "*.flac" | head -1)
    if [[ -n "$sample_audio" ]]; then
        if command -v soxi >/dev/null 2>&1; then
            local sample_info=$(soxi "$sample_audio" 2>/dev/null)
            if [[ $? -eq 0 ]]; then
                print_success "Audio files appear to be valid FLAC format"
            else
                print_warning "Sample audio file may be corrupted: $sample_audio"
            fi
        fi
    fi
    
    return 0
}

# Function to create a simple test script
create_test_script() {
    local data_dir=$1
    local script_path="${data_dir}/test_librispeech.sh"
    
    cat > "$script_path" << 'EOF'
#!/bin/bash
# Quick test script for LibriSpeech dataset

LIBRISPEECH_DIR="$1"
if [[ -z "$LIBRISPEECH_DIR" ]]; then
    LIBRISPEECH_DIR="$(dirname "$0")/LibriSpeech"
fi

echo "Testing LibriSpeech dataset at: $LIBRISPEECH_DIR"

if [[ ! -d "$LIBRISPEECH_DIR" ]]; then
    echo "Error: LibriSpeech directory not found"
    exit 1
fi

echo "Available subsets:"
ls -1 "$LIBRISPEECH_DIR"

echo ""
echo "Sample files from train-clean-100:"
find "$LIBRISPEECH_DIR/train-clean-100" -name "*.flac" | head -3

echo ""
echo "Sample transcript:"
find "$LIBRISPEECH_DIR/train-clean-100" -name "*.trans.txt" | head -1 | xargs head -3

echo ""
echo "Dataset appears ready for training!"
EOF
    
    chmod +x "$script_path"
    print_success "Created test script: $script_path"
}

# Parse command line arguments
DATA_DIR="$DEFAULT_DATA_DIR"
SUBSETS="$DEFAULT_SUBSETS"
KEEP_ARCHIVES=false
FORCE=false

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
            KEEP_ARCHIVES=true
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
            SUBSETS="train-clean-100 train-clean-360 dev-clean test-clean"
            shift
            ;;
        --minimal)
            SUBSETS="train-clean-100 dev-clean"
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

# Check dependencies
missing_deps=()
if ! command -v tar >/dev/null 2>&1; then
    missing_deps+=("tar")
fi
if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    missing_deps+=("wget or curl")
fi

if [[ ${#missing_deps[@]} -gt 0 ]]; then
    print_error "Missing required dependencies: ${missing_deps[*]}"
    exit 1
fi

# Print configuration
echo "=================================================="
echo "LibriSpeech Download Configuration"
echo "=================================================="
echo "Data directory: $DATA_DIR"
echo "Subsets to download: $SUBSETS"
echo "Keep archives: $KEEP_ARCHIVES"
echo "Force re-download: $FORCE"
echo "=================================================="
echo ""

# Estimate total download size
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
print_status "Make sure you have sufficient disk space and a stable internet connection"
echo ""

# Confirm before proceeding (skip if running non-interactively)
if [[ -t 0 ]]; then
    read -p "Proceed with download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Download cancelled"
        exit 0
    fi
else
    print_status "Non-interactive mode detected - proceeding with download automatically"
fi

# Start download process
print_status "Starting LibriSpeech download..."
start_time=$(date +%s)

# Download each subset
failed_subsets=()
for subset in $SUBSETS; do
    if ! download_subset "$subset" "$DATA_DIR" "$KEEP_ARCHIVES" "$FORCE"; then
        failed_subsets+=("$subset")
    fi
done

# Check for failures
if [[ ${#failed_subsets[@]} -gt 0 ]]; then
    print_error "Failed to download subsets: ${failed_subsets[*]}"
    exit 1
fi

# Verify dataset
if ! verify_dataset "$DATA_DIR" "$SUBSETS"; then
    print_error "Dataset verification failed"
    exit 1
fi

# Create test script
create_test_script "$DATA_DIR"

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo ""
echo "=================================================="
print_success "LibriSpeech download completed successfully!"
echo "=================================================="
print_success "Downloaded to: $DATA_DIR/LibriSpeech"
print_success "Time taken: ${hours}h ${minutes}m ${seconds}s"
print_success "Subsets downloaded: $SUBSETS"
echo ""
print_status "Next steps:"
echo "  1. Test the dataset:"
echo "     $DATA_DIR/test_librispeech.sh"
echo ""
echo "  2. Start training:"
echo "     cd $(dirname "$0")/.."
echo "     ./examples/train_librispeech.sh --data-dir $DATA_DIR/LibriSpeech"
echo ""
echo "  3. Or use custom training parameters:"
echo "     ./examples/train_librispeech.sh \\"
echo "       --data-dir $DATA_DIR/LibriSpeech \\"
echo "       --batch-size 16 \\"
echo "       --epochs 30 \\"
echo "       --train-subset train-clean-360"
echo "=================================================="
