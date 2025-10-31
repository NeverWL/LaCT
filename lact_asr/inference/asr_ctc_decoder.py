#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR Inference with CTC Beam Search Decoder using torchaudio's decoder.
Integrates KenLM for language model support following torchaudio's tutorial.
"""

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
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files
import numpy as np
from jiwer import wer, cer

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lact_asr_model import (
    LaCTASRConfig, 
    LaCTASRForCTC, 
)
from data import ASRDataset, ASRDataCollator, create_asr_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRCTCInference:
    """
    Inference class for LaCT ASR models with torchaudio's CTC decoder.
    Supports KenLM language model and lexicon-constrained decoding.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = 'cuda',
        # CTC decoder parameters
        tokens: Optional[List[str]] = None,
        lexicon: Optional[str] = None,
        lm: Optional[str] = None,
        nbest: int = 3,
        beam_size: int = 1500,
        beam_size_token: Optional[int] = None,
        beam_threshold: int = 25,
        lm_weight: float = 3.23,
        word_score: float = -0.26,
        unk_score: float = float('-inf'),
        sil_score: float = 0.0,
        log_add: bool = False,
        blank_token: str = "-",
        # Use pretrained LibriSpeech files if available
        use_pretrained_librispeech: bool = True,
    ):
        """
        Initialize ASR CTC inference with decoder.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model config file
            device: Device for inference
            tokens: List of tokens (characters) the model predicts
            lexicon: Path to lexicon file (word -> tokens mapping)
            lm: Path to KenLM language model (.arpa or .bin)
            nbest: Number of best hypotheses to return
            beam_size: Beam size for decoding
            beam_size_token: Number of tokens to consider per hypothesis
            beam_threshold: Pruning threshold for beam search
            lm_weight: Weight for language model score
            word_score: Score added when finishing a word
            unk_score: Score for unknown words
            sil_score: Score for silence tokens
            log_add: Use log add for lexicon Trie smearing
            blank_token: CTC blank token character
            use_pretrained_librispeech: Automatically download LibriSpeech decoder files if available
        """
        self.device = device
        self.blank_token = blank_token
        
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
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Vocabulary size: {self.config.ctc_vocab_size}")
        
        # Setup decoder parameters
        if use_pretrained_librispeech and tokens is None:
            # Try to download pretrained files
            try:
                logger.info("Attempting to download pretrained LibriSpeech decoder files...")
                files = download_pretrained_files("librispeech-4-gram")
                tokens = files.tokens
                if lexicon is None:
                    lexicon = files.lexicon
                if lm is None:
                    lm = files.lm
                logger.info("✓ Using pretrained LibriSpeech decoder files")
            except Exception as e:
                logger.warning(f"Could not download pretrained files: {e}")
                if tokens is None:
                    raise ValueError("Must provide tokens or enable use_pretrained_librispeech with internet access")
        
        if tokens is None:
            raise ValueError("Must provide tokens list")
        
        # Create CTC decoder
        decoder_kwargs = {
            'tokens': tokens,
            'nbest': nbest,
            'beam_size': beam_size,
            'beam_threshold': beam_threshold,
            'lm_weight': lm_weight,
            'word_score': word_score,
            'unk_score': unk_score,
            'sil_score': sil_score,
            'log_add': log_add,
        }
        
        if lexicon:
            decoder_kwargs['lexicon'] = lexicon
            logger.info(f"Lexicon: {lexicon}")
        
        if lm:
            decoder_kwargs['lm'] = lm
            logger.info(f"Language Model: {lm}")
        else:
            logger.info("No language model provided - using CTC only")
        
        if beam_size_token is not None:
            decoder_kwargs['beam_size_token'] = beam_size_token
        
        self.decoder = ctc_decoder(**decoder_kwargs)
        
        logger.info(f"CTC Decoder initialized:")
        logger.info(f"  Beam size: {beam_size}")
        logger.info(f"  LM weight: {lm_weight}")
        logger.info(f"  Word score: {word_score}")
        logger.info(f"  N-best: {nbest}")
    
    def transcribe_file(
        self, 
        audio_path: str, 
        sample_rate: Optional[int] = None
    ) -> Dict[str, Union[str, List[str], float]]:
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
    ) -> Dict[str, Union[str, List[str], float]]:
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
            emission = torch.log_softmax(logits, dim=-1)
        
        # Decode with CTC decoder
        hypotheses = self.decoder(emission)
        
        # Get best hypothesis
        if hypotheses[0]:
            best_hyp = hypotheses[0][0]
            transcript = " ".join(best_hyp.words).strip()
            all_nbest = [" ".join(hyp.words).strip() for hyp in hypotheses[0]]
        else:
            transcript = ""
            all_nbest = []
        
        return {
            'text': transcript,
            'nbest': all_nbest,
            'score': best_hyp.score if hypotheses[0] else 0.0,
            'audio_length': waveform.shape[0] / self.config.sample_rate,
        }
    
    def transcribe_batch(
        self, 
        audio_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, Union[str, List[str], float]]]:
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
            
            # Decode each sample individually (CTC decoder expects full emissions)
            for j, (path, length) in enumerate(zip(batch_paths, batch_lengths)):
                # Get emission for this sample
                sample_logits = logits[j]
                emission = torch.log_softmax(sample_logits.unsqueeze(0), dim=-1)
                
                # Decode
                hypotheses = self.decoder(emission)
                
                if hypotheses[0]:
                    best_hyp = hypotheses[0][0]
                    transcript = " ".join(best_hyp.words).strip()
                    all_nbest = [" ".join(hyp.words).strip() for hyp in hypotheses[0]]
                    score = best_hyp.score
                else:
                    transcript = ""
                    all_nbest = []
                    score = 0.0
                
                results.append({
                    'text': transcript,
                    'nbest': all_nbest,
                    'score': score,
                    'audio_length': length / self.config.sample_rate,
                    'audio_path': path
                })
        
        return results


class ASRCTCEvaluator:
    """
    Evaluator for ASR models using CTC decoder with WER/CER metrics.
    """
    
    def __init__(
        self,
        inference: ASRCTCInference,
    ):
        self.inference = inference
    
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
        collator = ASRDataCollator(hop_length=self.inference.config.hop_length)
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
            
            # Decode each sample
            for i in range(len(batch['audio_input'])):
                if max_samples and num_processed >= max_samples:
                    break
                
                # Get sample emission
                sample_logits = logits[i].unsqueeze(0)  # [1, time, vocab]
                emission = torch.log_softmax(sample_logits, dim=-1)
                
                # Decode with CTC decoder
                hypotheses = self.inference.decoder(emission)
                
                if hypotheses[0]:
                    pred_text = " ".join(hypotheses[0][0].words).strip().lower()
                else:
                    pred_text = ""
                
                # Get reference text
                ref_text = batch['texts'][i].lower()
                
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
    parser = argparse.ArgumentParser(description="ASR CTC Inference and Evaluation")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config_path", type=str, help="Path to model config file")
    
    # Decoder arguments
    parser.add_argument("--tokens", type=str, help="Path to tokens file or comma-separated list")
    parser.add_argument("--lexicon", type=str, help="Path to lexicon file")
    parser.add_argument("--lm", type=str, help="Path to KenLM language model")
    parser.add_argument("--use_pretrained_librispeech", action="store_true", 
                       help="Use pretrained LibriSpeech decoder files")
    parser.add_argument("--beam_size", type=int, default=1500, help="Beam size for decoding")
    parser.add_argument("--beam_size_token", type=int, help="Number of tokens to consider per hypothesis")
    parser.add_argument("--beam_threshold", type=int, default=25, help="Beam threshold")
    parser.add_argument("--lm_weight", type=float, default=3.23, help="LM weight")
    parser.add_argument("--word_score", type=float, default=-0.26, help="Word score")
    parser.add_argument("--nbest", type=int, default=3, help="Number of best hypotheses")
    
    # Inference arguments
    parser.add_argument("--mode", type=str, choices=["transcribe", "evaluate"], default="transcribe",
                       help="Mode: transcribe audio files or evaluate on dataset")
    parser.add_argument("--audio_file", type=str, help="Single audio file to transcribe")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--output_file", type=str, help="Output file for transcriptions")
    
    # Evaluation arguments
    parser.add_argument("--test_manifest", type=str, help="Test manifest file for evaluation")
    parser.add_argument("--test_data_dir", type=str, help="Test dataset directory (LibriSpeech)")
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
    
    # Parse tokens if provided as file or comma-separated
    tokens = None
    if args.tokens:
        if os.path.exists(args.tokens):
            with open(args.tokens, 'r') as f:
                tokens = [line.strip() for line in f if line.strip()]
        else:
            tokens = [t.strip() for t in args.tokens.split(',')]
    
    # Initialize inference
    inference = ASRCTCInference(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        tokens=tokens,
        lexicon=args.lexicon,
        lm=args.lm,
        beam_size=args.beam_size,
        beam_size_token=args.beam_size_token,
        beam_threshold=args.beam_threshold,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        nbest=args.nbest,
        use_pretrained_librispeech=args.use_pretrained_librispeech,
    )
    
    if args.mode == "transcribe":
        if args.audio_file:
            # Transcribe single file
            result = inference.transcribe_file(args.audio_file)
            print(f"Transcription: {result['text']}")
            print(f"Score: {result['score']:.2f}")
            
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
        evaluator = ASRCTCEvaluator(inference)
        
        if args.test_manifest:
            # Evaluate on manifest
            test_dataset = ASRDataset(
                manifest_path=args.test_manifest,
                sample_rate=inference.config.sample_rate,
                normalize_text=True
            )
            
            metrics = evaluator.evaluate_dataset(
                test_dataset,
                batch_size=args.batch_size,
                max_samples=args.max_eval_samples
            )
            
        elif args.test_data_dir and args.test_subset:
            # Evaluate on LibriSpeech
            from data import LibriSpeechDataset
            test_dataset = LibriSpeechDataset(
                root_dir=args.test_data_dir,
                subset=args.test_subset,
                sample_rate=inference.config.sample_rate,
                max_duration=30.0,
                normalize_text=True,
            )
            
            metrics = evaluator.evaluate_dataset(
                test_dataset,
                batch_size=args.batch_size,
                max_samples=args.max_eval_samples
            )
        else:
            logger.error("Please provide --test_manifest or --test_data_dir and --test_subset")
            return
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"WER: {metrics['wer']:.3f}")
        print(f"CER: {metrics['cer']:.3f}")
        print(f"Samples: {metrics['num_samples']}")
        print(f"Total audio duration: {metrics['total_audio_duration']:.1f}s")


if __name__ == "__main__":
    main()

