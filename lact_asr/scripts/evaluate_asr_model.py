#!/usr/bin/env python3
"""
Comprehensive evaluation script for LaCT and Wav2Vec2 ASR models.
Computes WER, CER on test sets and generates detailed analysis.
"""

import sys
import argparse
import torch
from pathlib import Path
import json
import time
from jiwer import wer, cer
from collections import defaultdict
import numpy as np
from types import SimpleNamespace

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
except ImportError:
    Wav2Vec2ForCTC = None
    Wav2Vec2Processor = None

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.asr_ctc_decoder import ASRCTCInference
from data import LibriSpeechDataset, ASRDataCollator, create_asr_dataloader


class Wav2Vec2InferenceAdapter:
    """
    Adapter to evaluate HuggingFace Wav2Vec2 style CTC models with this script.
    Provides a minimal interface similar to ASRCTCInference.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        revision: str | None = None,
        cache_dir: str | None = None,
    ):
        if Wav2Vec2ForCTC is None or Wav2Vec2Processor is None:
            raise ImportError(
                "transformers is required for Wav2Vec2 evaluation. "
                "Install it with `pip install transformers`."
            )

        self.model_id = model_id
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        ).to(device)
        self.model.eval()

        # Provide config-like namespace for downstream code
        self.config = SimpleNamespace(
            sample_rate=16000,
            hop_length=160,
            hidden_size=getattr(self.model.config, "hidden_size", "unknown"),
            num_hidden_layers=getattr(self.model.config, "num_hidden_layers", "unknown"),
            num_lact_heads="N/A",
            num_attn_heads=getattr(self.model.config, "num_attention_heads", "unknown"),
        )

    def _prepare_waveforms(self, batch, max_samples: int | None = None):
        """Trim padded waveforms back to their original lengths."""
        hop_length = self.config.hop_length
        waveforms = []
        sample_lengths = []

        for audio, length_frames in zip(batch["audio_input"], batch["input_lengths"]):
            sample_length = int(length_frames.item()) * hop_length
            waveform = audio[:sample_length].cpu()
            waveforms.append(waveform.numpy())
            sample_lengths.append(sample_length)

        if max_samples is not None:
            waveforms = waveforms[:max_samples]
            sample_lengths = sample_lengths[:max_samples]

        return waveforms, sample_lengths

    def infer_batch(self, batch, max_samples: int | None = None):
        """
        Run the Wav2Vec2 model on a batch and return normalized transcripts.
        """
        waveforms, sample_lengths = self._prepare_waveforms(batch, max_samples=max_samples)
        if not waveforms:
            return [], sample_lengths, 0.0

        start_time = time.time()
        inputs = self.processor(
            waveforms,
            sampling_rate=self.config.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        with torch.no_grad():
            logits = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            ).logits

        inference_time = time.time() - start_time

        predicted_ids = torch.argmax(logits, dim=-1)
        transcripts = self.processor.batch_decode(predicted_ids)
        transcripts = [self._normalize_text(t) for t in transcripts]

        return transcripts, sample_lengths, inference_time

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = " ".join(text.split())
        return text


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of LaCT ASR model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lact", "wav2vec2"],
        default="lact",
        help="Type of model to evaluate."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt file) or directory. "
             "For HuggingFace models this can point to a local clone."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to model config file (config.json). Required for LaCT models."
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="HuggingFace model identifier for Wav2Vec2 evaluation."
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Specific revision of the HuggingFace model to download."
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for HuggingFace downloads."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing LibriSpeech dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        nargs="+",
        default=["dev-clean", "test-clean"],
        help="Test sets to evaluate on (space-separated)"
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Beam width for CTC decoding (1 = greedy)"
    )
    parser.add_argument(
        "--lm-weight",
        type=float,
        default=3.23,
        help="Language model weight (default: 3.23)"
    )
    parser.add_argument(
        "--word-score",
        type=float,
        default=-0.26,
        help="Word insertion score (default: -0.26)"
    )
    parser.add_argument(
        "--beam-threshold",
        type=int,
        default=25,
        help="Beam pruning threshold (default: 25)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per test set (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (cuda or cpu)"
    )
    parser.add_argument(
        "--move-emission-to-cpu",
        action="store_true",
        help="Move emission tensors to CPU before CTC decoding (required by some decoders)"
    )
    
    return parser.parse_args()


def evaluate_test_set(
    inference,
    test_set,
    data_dir,
    max_samples,
    batch_size,
    device,
    move_emission_to_cpu=False,
    model_type: str = "lact",
    wav2vec2_adapter: Wav2Vec2InferenceAdapter | None = None,
):
    """
    Evaluate model on a single test set.
    
    Args:
        inference: ASRCTCInference instance
        test_set: Name of test set (e.g., 'dev-clean')
        data_dir: Path to LibriSpeech data directory
        max_samples: Maximum number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation
        device: Device to run on
        move_emission_to_cpu: Whether to move emissions to CPU before decoding
    
    Returns:
        dict: Evaluation results including WER, CER, timing info, etc.
    """
    print(f"\n{'=' * 80}")
    print(f"Evaluating on {test_set}")
    print(f"{'=' * 80}")
    
    if model_type == "lact":
        if inference is None:
            raise ValueError("ASRCTCInference instance is required for LaCT evaluation.")
        model_config = inference.config
    elif model_type == "wav2vec2":
        if wav2vec2_adapter is None:
            raise ValueError("Wav2Vec2InferenceAdapter instance is required for Wav2Vec2 evaluation.")
        model_config = wav2vec2_adapter.config
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load test dataset
    test_dataset = LibriSpeechDataset(
        root_dir=data_dir,
        subset=test_set,
        sample_rate=model_config.sample_rate,
        max_duration=30.0,
        normalize_text=True,
    )
    
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create dataloader
    collator = ASRDataCollator(hop_length=model_config.hop_length)
    test_dataloader = create_asr_dataloader(
        test_dataset,
        batch_size=batch_size,
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
    
    max_samples_val = max_samples if max_samples is not None else 999999
    num_processed = 0
    
    print(f"\nRunning inference...")
    
    if model_type == "lact":
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if num_processed >= max_samples_val:
                break
            
            # Move to device
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            
            # Time inference
            start_time = time.time()
            
            outputs = inference.model(
                audio_input=batch['audio_input'],
                input_lengths=batch['input_lengths']
            )
            logits = outputs.logits
            
            inference_time = time.time() - start_time
            
            # Decode each sample with CTC decoder
            for i in range(len(batch['audio_input'])):
                if num_processed >= max_samples_val:
                    break
                
                # Get sample emission
                sample_logits = logits[i].unsqueeze(0)  # [1, time, vocab]
                emission = torch.log_softmax(sample_logits, dim=-1)
                
                # Move emission to CPU for CTC decoder if requested
                if move_emission_to_cpu:
                    emission = emission.cpu()
                
                # Decode with CTC decoder
                hypotheses = inference.decoder(emission)
                
                if hypotheses[0]:
                    pred_text = " ".join(hypotheses[0][0].words).strip().lower()
                else:
                    pred_text = ""
                
                ref_text = batch['texts'][i].lower()
                
                all_predictions.append(pred_text)
                all_references.append(ref_text)
                
                # Track metrics
                    audio_dur = len(batch['audio_input'][i]) / model_config.sample_rate
                audio_durations.append(audio_dur)
                inference_times.append(inference_time / len(batch['audio_input']))
                
                # Track errors by audio length
                sample_wer = wer(ref_text, pred_text)
                length_bucket = int(audio_dur // 5) * 5  # 0-5s, 5-10s, etc.
                    errors_by_length[length_bucket].append(sample_wer)
                    
                    num_processed += 1
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Processed {num_processed}/{min(max_samples_val, len(test_dataset))} samples...")
    else:  # Wav2Vec2 path
        for batch_idx, batch in enumerate(test_dataloader):
            if num_processed >= max_samples_val:
                break

            remaining = max_samples_val - num_processed
            predictions, sample_lengths, inference_time = wav2vec2_adapter.infer_batch(
                batch,
                max_samples=remaining
            )

            references = [text.lower() for text in batch["texts"][:len(predictions)]]

            for pred_text, ref_text, sample_length in zip(predictions, references, sample_lengths):
                audio_dur = sample_length / model_config.sample_rate
                inference_times.append(inference_time / max(len(predictions), 1))
                audio_durations.append(audio_dur)

                all_predictions.append(pred_text)
                all_references.append(ref_text)

                sample_wer = wer(ref_text, pred_text)
                length_bucket = int(audio_dur // 5) * 5
                errors_by_length[length_bucket].append(sample_wer)
                
                num_processed += 1
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {num_processed}/{min(max_samples_val, len(test_dataset))} samples...")
    
    # Compute metrics
    print(f"\n{'=' * 80}")
    print(f"Results for {test_set}")
    print(f"{'=' * 80}")
    
    overall_wer = wer(all_references, all_predictions) * 100 if all_predictions else 0.0
    overall_cer = cer(all_references, all_predictions) * 100 if all_predictions else 0.0
    
    avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0
    total_audio_duration = sum(audio_durations)
    rtf = (sum(inference_times) / total_audio_duration) if total_audio_duration > 0 else 0.0  # Real-Time Factor
    
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
    
    # Return results
    return {
        'wer': overall_wer,
        'cer': overall_cer,
        'num_samples': num_processed,
        'avg_inference_time_ms': avg_inference_time * 1000,
        'rtf': rtf,
        'total_audio_hours': total_audio_duration / 3600,
        'wer_by_length': {str(k): float(np.mean(v) * 100) for k, v in errors_by_length.items()}
    }


def save_results(
    all_results,
    model_path,
    config_path,
    output_dir,
    beam_width,
    lm_weight,
    word_score,
    beam_threshold,
    inference_config
):
    """Save evaluation results to JSON and markdown files."""
    print(f"\n{'=' * 80}")
    print(f"Saving Results")
    print(f"{'=' * 80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model_checkpoint': model_path,
            'config': config_path,
            'training_step': 'unknown',
            'training_epoch': 'unknown',
            'model_config': {
                'hidden_size': inference_config.hidden_size,
                'num_layers': inference_config.num_hidden_layers,
                'num_lact_heads': inference_config.num_lact_heads,
                'num_attn_heads': inference_config.num_attn_heads,
            },
            'decoder_config': {
                'beam_width': beam_width,
                'lm_weight': lm_weight,
                'word_score': word_score,
                'beam_threshold': beam_threshold,
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"✓ Saved JSON results to: {results_file}")
    
    # Save markdown report
    report_file = output_dir / "evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write(f"# LaCT ASR Evaluation Report\n\n")
        f.write(f"**Model:** {inference_config.hidden_size}d hidden, {inference_config.num_hidden_layers} layers, {inference_config.num_lact_heads} LaCT heads\n")
        f.write(f"**Training:** Step unknown, Epoch unknown\n")
        f.write(f"**Decoder:** Beam size={beam_width}, LM weight={lm_weight}, Word score={word_score}, Beam threshold={beam_threshold}\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Results Summary\n\n")
        f.write(f"| Test Set | WER | CER | Samples | RTF |\n")
        f.write(f"|----------|-----|-----|---------|-----|\n")
        
        for test_set, results in all_results.items():
            f.write(f"| {test_set} | {results['wer']:.2f}% | {results['cer']:.2f}% | ")
            f.write(f"{results['num_samples']} | {results['rtf']:.3f}x |\n")
        
        f.write(f"\n## Detailed Results\n\n")
        for test_set, results in all_results.items():
            f.write(f"### {test_set}\n\n")
            f.write(f"- **WER:** {results['wer']:.2f}%\n")
            f.write(f"- **CER:** {results['cer']:.2f}%\n")
            f.write(f"- **Samples:** {results['num_samples']}\n")
            f.write(f"- **Avg inference time:** {results['avg_inference_time_ms']:.1f}ms\n")
            f.write(f"- **Real-Time Factor:** {results['rtf']:.3f}x\n")
            f.write(f"- **Total audio:** {results['total_audio_hours']:.2f} hours\n\n")
    
    print(f"✓ Saved report to: {report_file}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("=" * 80)
    print("LaCT ASR Comprehensive Evaluation with CTC Beam Search")
    print("=" * 80)
    
    # Setup decoder parameters (only relevant for LaCT)
    use_beam_search = args.beam_width > 1
    
    print(f"\nDecoder Configuration:")
    print(f"  Beam width: {args.beam_width}")
    print(f"  Beam search: {'Enabled' if use_beam_search else 'Disabled (greedy)'}")
    print(f"  LM weight: {args.lm_weight}")
    print(f"  Word score: {args.word_score}")
    print(f"  Beam threshold: {args.beam_threshold}")
    
    inference = None
    wav2vec2_adapter = None
    model_identifier = args.model_path
    config_identifier = args.config_path
    
    if args.model_type == "lact":
        if not args.model_path or not args.config_path:
            raise ValueError("--model-path and --config-path are required for LaCT models.")
        
        print(f"\nInitializing LaCT CTC inference...")
    inference = ASRCTCInference(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        use_pretrained_librispeech=True,  # Use LibriSpeech decoder files
        beam_size=args.beam_width if use_beam_search else 1,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        beam_threshold=args.beam_threshold,
        nbest=1,
    )
        print(f"✓ LaCT CTC inference initialized")
    else:
        if Wav2Vec2ForCTC is None or Wav2Vec2Processor is None:
            raise ImportError(
                "transformers is not installed. Install it with `pip install transformers` "
                "to evaluate Wav2Vec2 models."
            )
        model_identifier = args.model_path if args.model_path else args.hf_model_id
        config_identifier = args.hf_model_id
        
        if args.beam_width > 1:
            print("⚠️  Beam search parameters are ignored for Wav2Vec2 evaluation (greedy decoding only).")
        if args.move_emission_to_cpu:
            print("⚠️  --move-emission-to-cpu is not required for Wav2Vec2 models and will be ignored.")
        
        print(f"\nInitializing Wav2Vec2 model ({model_identifier})...")
        wav2vec2_adapter = Wav2Vec2InferenceAdapter(
            model_id=model_identifier,
            device=args.device,
            revision=args.hf_revision,
            cache_dir=args.hf_cache_dir,
        )
        print("✓ Wav2Vec2 model loaded")
    
    # Evaluation results storage
    all_results = {}
    
    # Evaluate on each test set
    for test_set in args.test_sets:
        results = evaluate_test_set(
            inference=inference,
            test_set=test_set,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            device=args.device,
            move_emission_to_cpu=args.move_emission_to_cpu if args.model_type == "lact" else False,
            model_type=args.model_type,
            wav2vec2_adapter=wav2vec2_adapter
        )
        all_results[test_set] = results
    
    # Determine config for saving
    inference_config = inference.config if args.model_type == "lact" else wav2vec2_adapter.config
    
    # Save results
    save_results(
        all_results=all_results,
        model_path=model_identifier,
        config_path=config_identifier,
        output_dir=args.output_dir,
        beam_width=args.beam_width,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        beam_threshold=args.beam_threshold,
        inference_config=inference_config
    )
    
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


if __name__ == "__main__":
    main()

