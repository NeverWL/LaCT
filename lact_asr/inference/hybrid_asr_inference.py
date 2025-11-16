# -*- coding: utf-8 -*-

"""
Inference script for Hybrid LaCT + Wav2Vec2 ASR model with Test-Time Training (TTT) support.

This script provides:
- CTC decoding with beam search (using torchaudio's CTC decoder)
- TTT-based inference for improved robustness
- Support for LibriSpeech dataset evaluation
"""

import sys
import argparse
import torch
import torchaudio
from pathlib import Path
import json
import time
import logging
from typing import Optional, List, Dict, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from lact_asr_model import (
        HybridLaCTWav2Vec2Config,
        HybridLaCTWav2Vec2ForCTC,
    )
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent / "lact_asr_model"))
    from configuration_hybrid_asr import HybridLaCTWav2Vec2Config
    from modeling_hybrid_asr import HybridLaCTWav2Vec2ForCTC

from inference.ttt_procedure import TTTProcedure, create_ttt_procedure
from data import LibriSpeechDataset, ASRDataCollator, create_asr_dataloader

logger = logging.getLogger(__name__)


class HybridASRInference:
    """
    Inference class for Hybrid LaCT + Wav2Vec2 ASR model with TTT support.
    """
    
    def __init__(
        self,
        model: HybridLaCTWav2Vec2ForCTC,
        config: HybridLaCTWav2Vec2Config,
        device: str = "cuda",
        use_ttt: bool = True,
        beam_size: int = 50,
        lm_weight: float = 2.0,
        word_score: float = -1.0,
        beam_threshold: float = 100.0,
        move_emission_to_cpu: bool = True,
    ):
        """
        Initialize Hybrid ASR Inference.
        
        Args:
            model: HybridLaCTWav2Vec2ForCTC model
            config: HybridLaCTWav2Vec2Config configuration
            device: Device to run inference on
            use_ttt: Whether to use Test-Time Training
            beam_size: Beam size for CTC decoding
            lm_weight: Language model weight for beam search
            word_score: Word score bonus for beam search
            beam_threshold: Beam threshold for pruning
            move_emission_to_cpu: Move emissions to CPU before decoding (required for flashlight-text)
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_ttt = use_ttt
        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.word_score = word_score
        self.beam_threshold = beam_threshold
        self.move_emission_to_cpu = move_emission_to_cpu
        
        self.model.eval()
        
        # LaCT blocks already include TTT internally; no external orchestration needed
        self.ttt_procedure = None
        
        # Initialize CTC decoder if beam_size > 1
        self.ctc_decoder = None
        if beam_size > 1:
            try:
                from torchaudio.models.decoder import download_pretrained_files
                from torchaudio.models.decoder import ctc_decoder
                
                # Download pretrained decoder files
                decoder_files = download_pretrained_files("librispeech-4-gram")
                
                self.ctc_decoder = ctc_decoder(
                    lexicon=decoder_files.lexicon,
                    tokens=decoder_files.tokens,
                    lm=decoder_files.lm,
                    beam_size=beam_size,
                    lm_weight=lm_weight,
                    word_score=word_score,
                    beam_threshold=beam_threshold,
                )
                logger.info(f"CTC decoder initialized with beam_size={beam_size}")
            except Exception as e:
                logger.warning(f"Failed to initialize CTC decoder: {e}. Falling back to greedy decoding.")
                self.beam_size = 1
                self.ctc_decoder = None
    
    def transcribe_file(
        self,
        audio_path: str,
        sample_rate: int = 16000,
    ) -> str:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            sample_rate: Sample rate of audio (will resample if different)
            
        Returns:
            Transcription text
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Transcribe
        transcript = self.transcribe_audio(waveform.squeeze(0))
        
        return transcript
    
    def transcribe_audio(
        self,
        waveform: torch.Tensor,
    ) -> str:
        """
        Transcribe raw audio waveform.
        
        Args:
            waveform: Audio waveform [num_samples] or [batch, num_samples]
            
        Returns:
            Transcription text (single string if batch=1, list if batch>1)
        """
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            return_single = True
        else:
            return_single = False
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Run inference; LaCT performs fast-weight TTT internally during forward
        with torch.no_grad():
            outputs = self.model(audio_input=waveform)
            logits = outputs.logits
        
        # Decode
        transcripts = self._decode_logits(logits)
        
        if return_single:
            return transcripts[0]
        else:
            return transcripts
    
    def transcribe_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_samples: Optional[int] = None,
    ) -> Tuple[List[str], List[int], float]:
        """
        Transcribe a batch of audio.
        
        Args:
            batch: Batch dictionary with 'audio_input' and 'input_lengths'
            max_samples: Maximum number of samples to process (for debugging)
            
        Returns:
            transcripts: List of transcriptions
            sample_lengths: List of audio lengths
            inference_time: Total inference time in seconds
        """
        waveforms = batch['audio_input']
        input_lengths = batch['input_lengths']
        
        # Trim to actual lengths
        sample_lengths = []
        trimmed_waveforms = []
        
        for waveform, length_frames in zip(waveforms, input_lengths):
            # Calculate actual length in samples
            # Assuming 16kHz and feature encoder downsampling factor
            ds_factor = self.config.encoder_target_ds_factor
            sample_length = int(length_frames.item()) * ds_factor * 160  # Approximate
            sample_length = min(sample_length, waveform.shape[0])
            
            trimmed = waveform[:sample_length].cpu()
            trimmed_waveforms.append(trimmed)
            sample_lengths.append(sample_length)
        
        if max_samples is not None:
            trimmed_waveforms = trimmed_waveforms[:max_samples]
            sample_lengths = sample_lengths[:max_samples]
        
        # Pad to same length
        max_length = max(w.shape[0] for w in trimmed_waveforms)
        padded_waveforms = torch.zeros(
            (len(trimmed_waveforms), max_length),
            dtype=trimmed_waveforms[0].dtype
        )
        for i, w in enumerate(trimmed_waveforms):
            padded_waveforms[i, :w.shape[0]] = w
        
        padded_waveforms = padded_waveforms.to(self.device)
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(audio_input=padded_waveforms)
            logits = outputs.logits
        
        inference_time = time.time() - start_time
        
        # Decode
        transcripts = self._decode_logits(logits)
        
        return transcripts, sample_lengths, inference_time
    
    def _decode_logits(
        self,
        logits: torch.Tensor,
    ) -> List[str]:
        """
        Decode logits to text transcriptions.
        
        Args:
            logits: CTC logits [batch, seq_len, vocab_size]
            
        Returns:
            List of transcriptions
        """
        # Get emission probabilities
        emissions = torch.log_softmax(logits, dim=-1)  # [B, T, V]
        
        # Move to CPU if required (for flashlight-text)
        if self.move_emission_to_cpu:
            emissions = emissions.cpu()
        
        if self.ctc_decoder is not None and self.beam_size > 1:
            # Beam search decoding
            transcripts = []
            for emission in emissions:
                # CTC decoder expects [T, V] format
                emission_t = emission.transpose(0, 1)
                result = self.ctc_decoder(emission_t)
                if result:
                    transcript = result[0].words
                    if transcript:
                        # Join words with spaces
                        transcript_text = " ".join(transcript)
                    else:
                        transcript_text = ""
                else:
                    transcript_text = ""
                transcripts.append(transcript_text)
            
            return transcripts
        else:
            # Greedy decoding
            predicted_ids = torch.argmax(emissions, dim=-1)  # [B, T]
            transcripts = []
            
            for pred_ids in predicted_ids:
                # CTC decode: remove blanks and consecutive duplicates
                decoded = []
                prev = None
                for idx in pred_ids:
                    idx = idx.item()
                    if idx != self.config.ctc_blank_id and idx != prev:
                        decoded.append(idx)
                    prev = idx
                
                # Convert to text (placeholder - would need vocab mapping)
                # For now, return as indices
                transcript = " ".join(str(idx) for idx in decoded)
                transcripts.append(transcript)
            
            return transcripts


def load_hybrid_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[HybridLaCTWav2Vec2ForCTC, HybridLaCTWav2Vec2Config]:
    """
    Load hybrid model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = HybridLaCTWav2Vec2Config(**config_dict)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model = HybridLaCTWav2Vec2ForCTC(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Hybrid LaCT + Wav2Vec2 ASR Inference with TTT")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file (if different from checkpoint dir)")
    
    # Inference arguments
    parser.add_argument("--use_ttt", action="store_true", help="Use Test-Time Training")
    parser.add_argument("--beam_size", type=int, default=50, help="Beam size for decoding")
    parser.add_argument("--lm_weight", type=float, default=2.0, help="Language model weight")
    parser.add_argument("--word_score", type=float, default=-1.0, help="Word score bonus")
    parser.add_argument("--beam_threshold", type=float, default=100.0, help="Beam threshold")
    
    # Data arguments
    parser.add_argument("--audio_file", type=str, help="Path to single audio file")
    parser.add_argument("--data_dir", type=str, help="Path to LibriSpeech dataset")
    parser.add_argument("--subset", type=str, default="test-clean", help="Dataset subset")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--move_emission_to_cpu", action="store_true", 
                       help="Move emissions to CPU before decoding")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load model
    model, config = load_hybrid_model(args.checkpoint, args.device)
    
    # Create inference instance
    inference = HybridASRInference(
        model=model,
        config=config,
        device=args.device,
        use_ttt=args.use_ttt,
        beam_size=args.beam_size,
        lm_weight=args.lm_weight,
        word_score=args.word_score,
        beam_threshold=args.beam_threshold,
        move_emission_to_cpu=args.move_emission_to_cpu,
    )
    
    # Run inference
    if args.audio_file:
        # Single file inference
        transcript = inference.transcribe_file(args.audio_file)
        print(f"Transcription: {transcript}")
    
    elif args.data_dir:
        # Dataset evaluation
        from jiwer import wer, cer
        
        logger.info(f"Loading dataset: {args.subset}")
        dataset = LibriSpeechDataset(
            root_dir=args.data_dir,
            subset=args.subset,
            sample_rate=config.sample_rate,
            normalize_text=True,
        )
        
        # Reuse vocab from checkpoint if available
        # (would need to load from checkpoint or config)
        
        collator = ASRDataCollator()
        dataloader = create_asr_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        
        all_predictions = []
        all_references = []
        
        for batch_idx, batch in enumerate(dataloader):
            if args.max_samples and batch_idx >= args.max_samples:
                break
            
            transcripts, _, _ = inference.transcribe_batch(batch, max_samples=1)
            predictions = transcripts
            references = batch['texts']
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} samples")
        
        # Compute metrics
        wer_score = wer(all_references, all_predictions)
        cer_score = cer(all_references, all_predictions)
        
        logger.info(f"WER: {wer_score:.4f}")
        logger.info(f"CER: {cer_score:.4f}")
        
        print(f"\nResults:")
        print(f"WER: {wer_score:.4f}")
        print(f"CER: {cer_score:.4f}")
    
    else:
        parser.print_help()
        logger.error("Must specify either --audio_file or --data_dir")


if __name__ == "__main__":
    main()

