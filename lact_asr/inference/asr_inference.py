#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from jiwer import wer, cer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lact_asr_model import (
    LaCTASRConfig, 
    LaCTASRForCTC, 
    AudioFeatureExtractor,
    MelSpectrogramExtractor
)
from data import ASRDataset, ASRDataCollator, create_asr_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRInference:
    """
    Inference class for LaCT ASR models.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = 'cuda',
        beam_width: int = 1,
    ):
        self.device = device
        self.beam_width = beam_width
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = LaCTASRConfig(**config_dict)
        else:
            # Try to load config from model directory
            model_dir = Path(model_path).parent
            config_file = model_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                self.config = LaCTASRConfig(**config_dict)
            else:
                raise ValueError("Config file not found. Please provide config_path.")
        
        # Load model
        self.model = LaCTASRForCTC(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Load vocabulary
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        
        if vocab_path and os.path.exists(vocab_path):
            self._load_vocab(vocab_path)
        else:
            logger.warning("No vocabulary file provided. Text decoding will return token indices.")
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Vocabulary size: {self.config.ctc_vocab_size}")
    
    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f if line.strip()]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        logger.info(f"Loaded vocabulary with {len(self.vocab)} characters")
    
    def transcribe_file(
        self, 
        audio_path: str, 
        sample_rate: Optional[int] = None
    ) -> Dict[str, Union[str, List[int], float]]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Target sample rate (uses config default if None)
            
        Returns:
            Dictionary containing transcription results
        """
        # Load audio
        waveform, orig_sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        target_sr = sample_rate or self.config.sample_rate
        if orig_sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Transcribe
        return self.transcribe_audio(waveform.squeeze(0))
    
    def transcribe_audio(
        self, 
        waveform: torch.Tensor
    ) -> Dict[str, Union[str, List[int], float]]:
        """
        Transcribe audio waveform.
        
        Args:
            waveform: Audio waveform tensor [num_samples]
            
        Returns:
            Dictionary containing transcription results
        """
        # Prepare input
        audio_input = waveform.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(audio_input=audio_input)
            logits = outputs.logits  # [batch, time, vocab]
            log_probs = F.log_softmax(logits, dim=-1)
        
        # Decode
        if self.beam_width == 1:
            # Greedy decoding
            predicted_ids = torch.argmax(log_probs, dim=-1)
            confidence_scores = torch.max(F.softmax(logits, dim=-1), dim=-1)[0]
            avg_confidence = confidence_scores.mean().item()
        else:
            # Beam search decoding (simplified implementation)
            predicted_ids, avg_confidence = self._beam_search_decode(log_probs)
        
        # CTC decoding - remove blanks and consecutive duplicates
        decoded_ids = self._ctc_decode(predicted_ids.squeeze(0).cpu().numpy())
        
        # Convert to text if vocabulary is available
        if self.vocab:
            text = self._ids_to_text(decoded_ids)
        else:
            text = None
        
        return {
            'text': text,
            'token_ids': decoded_ids,
            'confidence': avg_confidence,
            'audio_length': waveform.shape[0] / self.config.sample_rate
        }
    
    def transcribe_batch(
        self, 
        audio_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, Union[str, List[int], float]]]:
        """
        Transcribe multiple audio files in batches.
        
        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for processing
            
        Returns:
            List of transcription results
        """
        results = []
        
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            
            # Load and prepare batch
            batch_waveforms = []
            batch_lengths = []
            
            for path in batch_paths:
                try:
                    waveform, orig_sr = torchaudio.load(path)
                    
                    # Resample if necessary
                    if orig_sr != self.config.sample_rate:
                        resampler = torchaudio.transforms.Resample(orig_sr, self.config.sample_rate)
                        waveform = resampler(waveform)
                    
                    # Convert to mono
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    batch_waveforms.append(waveform.squeeze(0))
                    batch_lengths.append(waveform.shape[1])
                    
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
                    # Add placeholder for failed files
                    batch_waveforms.append(torch.zeros(1600))  # 0.1 seconds of silence
                    batch_lengths.append(1600)
            
            # Pad batch
            max_length = max(batch_lengths)
            padded_batch = []
            
            for waveform, length in zip(batch_waveforms, batch_lengths):
                if length < max_length:
                    padding = torch.zeros(max_length - length)
                    waveform = torch.cat([waveform, padding])
                padded_batch.append(waveform)
            
            # Process batch
            batch_tensor = torch.stack(padded_batch).to(self.device)
            input_lengths = torch.tensor(batch_lengths, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    audio_input=batch_tensor,
                    input_lengths=input_lengths
                )
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Decode each sample in the batch
            for j, (path, length) in enumerate(zip(batch_paths, batch_lengths)):
                sample_logits = logits[j, :length // (self.config.hop_length or 160)]
                sample_log_probs = log_probs[j, :length // (self.config.hop_length or 160)]
                
                if self.beam_width == 1:
                    predicted_ids = torch.argmax(sample_log_probs, dim=-1)
                    confidence_scores = torch.max(F.softmax(sample_logits, dim=-1), dim=-1)[0]
                    avg_confidence = confidence_scores.mean().item()
                else:
                    predicted_ids, avg_confidence = self._beam_search_decode(sample_log_probs.unsqueeze(0))
                    predicted_ids = predicted_ids.squeeze(0)
                
                # CTC decode
                decoded_ids = self._ctc_decode(predicted_ids.cpu().numpy())
                
                # Convert to text
                if self.vocab:
                    text = self._ids_to_text(decoded_ids)
                else:
                    text = None
                
                results.append({
                    'text': text,
                    'token_ids': decoded_ids,
                    'confidence': avg_confidence,
                    'audio_length': length / self.config.sample_rate,
                    'audio_path': path
                })
        
        return results
    
    def _ctc_decode(self, predictions: np.ndarray) -> List[int]:
        """
        CTC decoding - remove blanks and consecutive duplicates.
        
        Args:
            predictions: Array of predicted token IDs
            
        Returns:
            List of decoded token IDs
        """
        decoded = []
        prev = None
        
        for pred in predictions:
            if pred != 0 and pred != prev:  # 0 is blank token
                decoded.append(int(pred))
            prev = pred
        
        return decoded
    
    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs to text string."""
        if not self.idx_to_char:
            return str(token_ids)
        
        chars = []
        for token_id in token_ids:
            if token_id in self.idx_to_char:
                chars.append(self.idx_to_char[token_id])
        
        return ''.join(chars)
    
    def _beam_search_decode(
        self, 
        log_probs: torch.Tensor, 
        beam_width: Optional[int] = None
    ) -> tuple[torch.Tensor, float]:
        """
        Simple beam search decoding for CTC.
        
        Args:
            log_probs: Log probabilities [batch, time, vocab]
            beam_width: Beam width (uses self.beam_width if None)
            
        Returns:
            Tuple of (predicted_ids, average_confidence)
        """
        if beam_width is None:
            beam_width = self.beam_width
        
        batch_size, seq_len, vocab_size = log_probs.shape
        
        # For simplicity, this is a basic implementation
        # In practice, you'd want a more sophisticated beam search for CTC
        
        # For now, fall back to greedy decoding
        predicted_ids = torch.argmax(log_probs, dim=-1)
        probs = F.softmax(log_probs, dim=-1)
        confidence_scores = torch.max(probs, dim=-1)[0]
        avg_confidence = confidence_scores.mean().item()
        
        return predicted_ids, avg_confidence


class ASREvaluator:
    """
    Evaluator for ASR models that computes WER and CER metrics.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: str = 'cuda',
    ):
        self.inference = ASRInference(
            model_path=model_path,
            config_path=config_path,
            vocab_path=vocab_path,
            device=device
        )
    
    def evaluate_dataset(
        self,
        test_dataset: ASRDataset,
        batch_size: int = 8,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on a test dataset.
        
        Args:
            test_dataset: ASR test dataset
            batch_size: Batch size for evaluation
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create dataloader
        collator = ASRDataCollator(pad_token_id=0)
        dataloader = create_asr_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collator
        )
        
        predictions = []
        references = []
        total_audio_duration = 0.0
        
        num_processed = 0
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and num_processed >= max_samples:
                break
            
            # Move batch to device
            batch = {k: v.to(self.inference.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.inference.model(
                    audio_input=batch['audio_input'],
                    input_lengths=batch['input_lengths']
                )
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Decode predictions for each sample in batch
            for i in range(len(batch['audio_input'])):
                if max_samples and num_processed >= max_samples:
                    break
                
                # Get sample-specific data
                sample_length = batch['input_lengths'][i].item()
                sample_log_probs = log_probs[i, :sample_length]
                
                # Decode prediction
                predicted_ids = torch.argmax(sample_log_probs, dim=-1)
                decoded_ids = self.inference._ctc_decode(predicted_ids.cpu().numpy())
                
                if self.inference.vocab:
                    pred_text = self.inference._ids_to_text(decoded_ids)
                else:
                    pred_text = ' '.join(map(str, decoded_ids))
                
                # Get reference text
                ref_text = batch['texts'][i]
                
                predictions.append(pred_text)
                references.append(ref_text)
                
                # Track audio duration
                audio_duration = batch['input_lengths'][i].item() / self.inference.config.sample_rate
                total_audio_duration += audio_duration
                
                num_processed += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {num_processed} samples...")
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, references)
        metrics['total_audio_duration'] = total_audio_duration
        metrics['num_samples'] = len(predictions)
        metrics['rtf'] = 0.0  # Real-time factor (would need timing info)
        
        return metrics
    
    def evaluate_files(
        self,
        audio_files: List[str],
        reference_texts: List[str],
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate model on a list of audio files with reference transcripts.
        
        Args:
            audio_files: List of paths to audio files
            reference_texts: List of reference transcripts
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of evaluation metrics
        """
        assert len(audio_files) == len(reference_texts), "Number of audio files and references must match"
        
        logger.info(f"Evaluating on {len(audio_files)} files...")
        
        # Get predictions
        results = self.inference.transcribe_batch(audio_files, batch_size)
        
        predictions = []
        references = reference_texts
        total_audio_duration = sum(r['audio_length'] for r in results)
        
        for result in results:
            pred_text = result['text'] or ' '.join(map(str, result['token_ids']))
            predictions.append(pred_text)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, references)
        metrics['total_audio_duration'] = total_audio_duration
        metrics['num_samples'] = len(predictions)
        
        return metrics
    
    def _compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute WER and CER metrics."""
        
        # Filter out empty predictions/references
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        
        if not filtered_pairs:
            return {'wer': 1.0, 'cer': 1.0}
        
        filtered_predictions, filtered_references = zip(*filtered_pairs)
        
        # Compute WER (Word Error Rate)
        try:
            word_error_rate = wer(filtered_references, filtered_predictions)
        except Exception as e:
            logger.warning(f"Error computing WER: {e}")
            word_error_rate = 1.0
        
        # Compute CER (Character Error Rate)
        try:
            char_error_rate = cer(filtered_references, filtered_predictions)
        except Exception as e:
            logger.warning(f"Error computing CER: {e}")
            char_error_rate = 1.0
        
        return {
            'wer': word_error_rate,
            'cer': char_error_rate
        }


def main():
    parser = argparse.ArgumentParser(description="ASR Inference and Evaluation")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model config file")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    
    # Inference arguments
    parser.add_argument("--mode", type=str, choices=["transcribe", "evaluate"], default="transcribe",
                       help="Mode: transcribe audio files or evaluate on dataset")
    parser.add_argument("--audio_file", type=str, help="Single audio file to transcribe")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files to transcribe")
    parser.add_argument("--output_file", type=str, help="Output file for transcriptions")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for decoding")
    
    # Evaluation arguments
    parser.add_argument("--test_manifest", type=str, help="Test manifest file for evaluation")
    parser.add_argument("--test_dataset_type", type=str, choices=["generic", "librispeech", "commonvoice"],
                       default="generic", help="Type of test dataset")
    parser.add_argument("--test_data_dir", type=str, help="Test dataset directory")
    parser.add_argument("--test_subset", type=str, help="Test subset (for LibriSpeech)")
    parser.add_argument("--max_eval_samples", type=int, help="Maximum samples for evaluation")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    if args.mode == "transcribe":
        # Initialize inference
        inference = ASRInference(
            model_path=args.model_path,
            config_path=args.config_path,
            vocab_path=args.vocab_path,
            device=args.device,
            beam_width=args.beam_width
        )
        
        if args.audio_file:
            # Transcribe single file
            result = inference.transcribe_file(args.audio_file)
            print(f"Transcription: {result['text']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
        elif args.audio_dir:
            # Transcribe directory
            audio_dir = Path(args.audio_dir)
            audio_files = []
            
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_files.extend(list(audio_dir.glob(f"*{ext}")))
                audio_files.extend(list(audio_dir.glob(f"*{ext.upper()}")))
            
            audio_files = [str(f) for f in audio_files]
            logger.info(f"Found {len(audio_files)} audio files")
            
            results = inference.transcribe_batch(audio_files, args.batch_size)
            
            # Save results
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                logger.info(f"Saved transcriptions to {args.output_file}")
            else:
                for result in results:
                    print(f"{result['audio_path']}: {result['text']}")
        
        else:
            logger.error("Please provide --audio_file or --audio_dir for transcription")
    
    elif args.mode == "evaluate":
        # Initialize evaluator
        evaluator = ASREvaluator(
            model_path=args.model_path,
            config_path=args.config_path,
            vocab_path=args.vocab_path,
            device=args.device
        )
        
        if args.test_manifest:
            # Evaluate on manifest
            test_dataset = ASRDataset(
                manifest_path=args.test_manifest,
                vocab_path=args.vocab_path,
                sample_rate=evaluator.inference.config.sample_rate,
                normalize_text=True
            )
            
            metrics = evaluator.evaluate_dataset(
                test_dataset,
                batch_size=args.batch_size,
                max_samples=args.max_eval_samples
            )
            
        else:
            logger.error("Please provide --test_manifest for evaluation")
            return
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"WER: {metrics['wer']:.3f}")
        print(f"CER: {metrics['cer']:.3f}")
        print(f"Samples: {metrics['num_samples']}")
        print(f"Total audio duration: {metrics['total_audio_duration']:.1f}s")


if __name__ == "__main__":
    main()
