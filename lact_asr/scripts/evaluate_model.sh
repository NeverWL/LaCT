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

show_usage() {
    cat << EOF
Usage: $0 [options]

Comprehensive evaluation of trained LaCT ASR model.
Computes WER, CER, and other metrics on test sets.

Options:
  -c, --checkpoint-dir DIR    Directory containing trained model (default: $DEFAULT_CHECKPOINT_DIR)
  -d, --data-dir DIR          Directory with LibriSpeech dataset (default: $DEFAULT_DATA_DIR)
  -o, --output-dir DIR        Directory for evaluation results (default: $DEFAULT_OUTPUT_DIR)
  --test-sets SETS            Space-separated test sets (default: "dev-clean test-clean")
  --beam-width WIDTH          Beam width for decoding (default: 1)
  --max-samples NUM           Max samples per test set (default: all)
  -h, --help                  Show this help message

Examples:
  $0                                              # Evaluate on dev-clean and test-clean
  $0 -c ./checkpoints/my_model                   # Evaluate specific checkpoint
  $0 --test-sets "dev-clean"                     # Evaluate only on dev-clean
  $0 dev-clean test-clean                        # Positional arguments for test sets
  $0 --beam-width 5                              # Use beam search decoding
  $0 --max-samples 500                           # Evaluate on first 500 samples

EOF
}

# Parse arguments
CHECKPOINT_DIR="$DEFAULT_CHECKPOINT_DIR"
DATA_DIR="$DEFAULT_DATA_DIR"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
TEST_SETS="dev-clean test-clean"
BEAM_WIDTH=1
MAX_SAMPLES=""

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
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
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
print_status "Data directory: $DATA_DIR"
print_status "Output directory: $OUTPUT_DIR"
print_status "Test sets: $TEST_SETS"
print_status "Beam width: $BEAM_WIDTH"
print_status "Max samples per set: ${MAX_SAMPLES:-all}"
echo ""

# Check if checkpoint exists
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    print_error "Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

MODEL_FILE="$CHECKPOINT_DIR/best_model.pt"
if [[ ! -f "$MODEL_FILE" ]]; then
    MODEL_FILE="$CHECKPOINT_DIR/latest_checkpoint.pt"
    if [[ ! -f "$MODEL_FILE" ]]; then
        print_error "No model checkpoint found in $CHECKPOINT_DIR"
        exit 1
    fi
    print_warning "Using latest_checkpoint.pt (best_model.pt not found)"
fi

CONFIG_FILE="$CHECKPOINT_DIR/config.json"
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

print_success "Found checkpoint: $MODEL_FILE"
print_success "Found config: $CONFIG_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comprehensive evaluation
print_status "Running comprehensive evaluation..."
echo ""
echo "=================================================="

python << EOF
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import time
from jiwer import wer, cer
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.append(str(Path.cwd()))

from lact_asr_model import LaCTASRConfig, LaCTASRForCTC
from data import LibriSpeechDataset, ASRDataCollator, create_asr_dataloader

print("=" * 80)
print("LaCT ASR Comprehensive Evaluation")
print("=" * 80)

# Load model
print(f"\nLoading model from: $MODEL_FILE")

with open("$CONFIG_FILE", 'r') as f:
    config_dict = json.load(f)
config = LaCTASRConfig(**config_dict)

checkpoint = torch.load("$MODEL_FILE", map_location='cuda')
model = LaCTASRForCTC(config)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    training_step = checkpoint.get('global_step', 'unknown')
    training_epoch = checkpoint.get('epoch', 'unknown')
else:
    model.load_state_dict(checkpoint)
    training_step = 'unknown'
    training_epoch = 'unknown'

model = model.to('cuda')
model.eval()

print(f"✓ Model loaded successfully")
print(f"  Training step: {training_step}")
print(f"  Training epoch: {training_epoch}")
print(f"  Model: {config.hidden_size}d, {config.num_hidden_layers} layers, {config.num_lact_heads} LaCT heads")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Evaluation results storage
all_results = {}

# Evaluate on each test set
test_sets = "$TEST_SETS".split()

for test_set in test_sets:
    print(f"\n{'=' * 80}")
    print(f"Evaluating on {test_set}")
    print(f"{'=' * 80}")
    
    # Load test dataset
    test_dataset = LibriSpeechDataset(
        root_dir="$DATA_DIR",
        subset=test_set,
        sample_rate=config.sample_rate,
        max_duration=30.0,
        normalize_text=True,
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create dataloader
    collator = ASRDataCollator(hop_length=config.hop_length)
    test_dataloader = create_asr_dataloader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )
    
    # Evaluation metrics
    all_predictions = []
    all_references = []
    inference_times = []
    audio_durations = []
    
    # Track errors by length
    errors_by_length = defaultdict(list)
    
    max_samples = ${MAX_SAMPLES:-999999}
    num_processed = 0
    
    print(f"\nRunning inference...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if num_processed >= max_samples:
                break
            
            # Move to device
            batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Time inference
            start_time = time.time()
            
            outputs = model(
                audio_input=batch['audio_input'],
                input_lengths=batch['input_lengths']
            )
            logits = outputs.logits
            
            inference_time = time.time() - start_time
            
            # Decode each sample
            for i in range(len(batch['audio_input'])):
                if num_processed >= max_samples:
                    break
                
                # Get sample data
                sample_length = batch['input_lengths'][i].item()
                sample_logits = logits[i, :sample_length]
                
                # Greedy decode
                predictions = torch.argmax(sample_logits, dim=-1)
                pred_indices = predictions.cpu().tolist()
                
                # CTC decode - remove blanks and duplicates
                decoded = []
                prev = None
                for idx in pred_indices:
                    if idx != 0 and idx != prev:
                        decoded.append(idx)
                    prev = idx
                
                # Convert to text
                pred_text = ''.join([test_dataset.idx_to_char.get(idx, '?') for idx in decoded])
                ref_text = batch['texts'][i]
                
                # Ensure both are lowercase for fair comparison
                pred_text = pred_text.lower().strip()
                ref_text = ref_text.lower().strip()
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                # Track metrics
                audio_dur = len(batch['audio_input'][i]) / config.sample_rate
                audio_durations.append(audio_dur)
                inference_times.append(inference_time / len(batch['audio_input']))
                
                # Track errors by audio length
                sample_wer = wer(ref_text, pred_text)
                length_bucket = int(audio_dur // 5) * 5  # 0-5s, 5-10s, etc.
                errors_by_length[length_bucket].append(sample_wer)
                
                num_processed += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {num_processed}/{min(max_samples, len(test_dataset))} samples...")
    
    # Compute metrics
    print(f"\n{'=' * 80}")
    print(f"Results for {test_set}")
    print(f"{'=' * 80}")
    
    overall_wer = wer(all_references, all_predictions) * 100
    overall_cer = cer(all_references, all_predictions) * 100
    
    avg_inference_time = np.mean(inference_times)
    total_audio_duration = sum(audio_durations)
    rtf = sum(inference_times) / total_audio_duration  # Real-Time Factor
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Samples evaluated: {num_processed}")
    print(f"  Word Error Rate (WER): {overall_wer:.2f}%")
    print(f"  Character Error Rate (CER): {overall_cer:.2f}%")
    print(f"  Average inference time: {avg_inference_time*1000:.1f}ms per sample")
    print(f"  Real-Time Factor (RTF): {rtf:.3f}x")
    print(f"  Total audio processed: {total_audio_duration/3600:.2f} hours")
    
    # WER by audio length
    print(f"\n📏 WER by Audio Duration:")
    for length_bucket in sorted(errors_by_length.keys()):
        bucket_wers = errors_by_length[length_bucket]
        avg_wer = np.mean(bucket_wers) * 100
        print(f"  {length_bucket:2d}-{length_bucket+5:2d}s: {avg_wer:5.2f}% WER ({len(bucket_wers)} samples)")
    
    # Show sample predictions
    print(f"\n📝 Sample Predictions (first 10):")
    print(f"{'=' * 80}")
    for i in range(min(10, len(all_predictions))):
        ref = all_references[i]
        pred = all_predictions[i]
        sample_wer = wer(ref, pred) * 100
        sample_cer = cer(ref, pred) * 100
        
        print(f"\nSample {i+1}:")
        print(f"  REF:  {ref}")
        print(f"  PRED: {pred}")
        print(f"  WER: {sample_wer:.1f}% | CER: {sample_cer:.1f}%")
    
    # Store results
    all_results[test_set] = {
        'wer': overall_wer,
        'cer': overall_cer,
        'num_samples': num_processed,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'rtf': rtf,
        'total_audio_hours': total_audio_duration / 3600,
        'wer_by_length': {str(k): float(np.mean(v) * 100) for k, v in errors_by_length.items()}
    }

# Save detailed results
print(f"\n{'=' * 80}")
print(f"Saving Results")
print(f"{'=' * 80}")

output_dir = Path("$OUTPUT_DIR")
output_dir.mkdir(parents=True, exist_ok=True)

# Save JSON results
results_file = output_dir / "evaluation_results.json"
with open(results_file, 'w') as f:
    json.dump({
        'model_checkpoint': "$MODEL_FILE",
        'config': "$CONFIG_FILE",
        'training_step': str(training_step),
        'training_epoch': str(training_epoch),
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_lact_heads': config.num_lact_heads,
            'num_attn_heads': config.num_attn_heads,
        },
        'beam_width': $BEAM_WIDTH,
        'results': all_results
    }, f, indent=2)

print(f"✓ Saved JSON results to: {results_file}")

# Save markdown report
report_file = output_dir / "evaluation_report.md"
with open(report_file, 'w') as f:
    f.write(f"# LaCT ASR Evaluation Report\\n\\n")
    f.write(f"**Model:** {config.hidden_size}d hidden, {config.num_hidden_layers} layers, {config.num_lact_heads} LaCT heads\\n")
    f.write(f"**Training:** Step {training_step}, Epoch {training_epoch}\\n")
    f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
    
    f.write(f"## Results Summary\\n\\n")
    f.write(f"| Test Set | WER | CER | Samples | RTF |\\n")
    f.write(f"|----------|-----|-----|---------|-----|\\n")
    
    for test_set, results in all_results.items():
        f.write(f"| {test_set} | {results['wer']:.2f}% | {results['cer']:.2f}% | ")
        f.write(f"{results['num_samples']} | {results['rtf']:.3f}x |\\n")
    
    f.write(f"\\n## Detailed Results\\n\\n")
    for test_set, results in all_results.items():
        f.write(f"### {test_set}\\n\\n")
        f.write(f"- **WER:** {results['wer']:.2f}%\\n")
        f.write(f"- **CER:** {results['cer']:.2f}%\\n")
        f.write(f"- **Samples:** {results['num_samples']}\\n")
        f.write(f"- **Avg inference time:** {results['avg_inference_time_ms']:.1f}ms\\n")
        f.write(f"- **Real-Time Factor:** {results['rtf']:.3f}x\\n")
        f.write(f"- **Total audio:** {results['total_audio_hours']:.2f} hours\\n\\n")

print(f"✓ Saved report to: {report_file}")

# Print summary
print(f"\n{'=' * 80}")
print(f"EVALUATION SUMMARY")
print(f"{'=' * 80}")

for test_set, results in all_results.items():
    print(f"\n{test_set}:")
    print(f"  WER: {results['wer']:.2f}%")
    print(f"  CER: {results['cer']:.2f}%")
    print(f"  RTF: {results['rtf']:.3f}x")

print(f"\n{'=' * 80}")

EOF

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

