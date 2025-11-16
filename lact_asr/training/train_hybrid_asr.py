#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for Hybrid LaCT + Wav2Vec2 ASR model.

This script extends train_asr.py with support for:
- Different learning rates for encoder (low LR) and transformer (normal LR)
- Wav2Vec2 pretrained encoder integration
- Test-Time Training (TTT) configuration
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchaudio
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import hybrid model components
try:
    from lact_asr_model import (
        HybridLaCTWav2Vec2Config,
        HybridLaCTWav2Vec2ForCTC,
    )
except ImportError:
    # Fallback if not in package
    sys.path.append(str(Path(__file__).parent.parent / "lact_asr_model"))
    from configuration_hybrid_asr import HybridLaCTWav2Vec2Config
    from modeling_hybrid_asr import HybridLaCTWav2Vec2ForCTC

# Import existing training utilities
from training.train_asr import ASRTrainer, create_datasets_and_dataloaders
from data import (
    ASRDataset, 
    LibriSpeechDataset, 
    CommonVoiceDataset,
    ASRDataCollator,
    create_asr_dataloader
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class HybridASRTrainer(ASRTrainer):
    """
    Trainer for Hybrid LaCT + Wav2Vec2 ASR model.
    
    Extends ASRTrainer with:
    - Separate learning rates for encoder and transformer
    - Proper handling of frozen/lightly-finetuned encoder
    """
    
    def __init__(
        self,
        config: HybridLaCTWav2Vec2Config,
        model: HybridLaCTWav2Vec2ForCTC,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        save_steps: int = 5000,
        eval_steps: int = 500,
        test_eval_steps: int = 5000,
        logging_steps: int = 100,
        max_steps: Optional[int] = None,
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        test_dataloader: Optional[DataLoader] = None,
        # Hybrid-specific parameters
        encoder_lr_scale: float = 0.1,  # Learning rate scale for encoder
        peak_lr: float = 5e-4,  # Peak learning rate
        warmup_steps: int = 10000,  # Warmup steps
    ):
        # Initialize parent class
        super().__init__(
            config=config,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            output_dir=output_dir,
            save_steps=save_steps,
            eval_steps=eval_steps,
            test_eval_steps=test_eval_steps,
            logging_steps=logging_steps,
            max_steps=max_steps,
            max_epochs=max_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            mixed_precision=mixed_precision,
            test_dataloader=test_dataloader,
        )
        
        self.encoder_lr_scale = encoder_lr_scale
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        
        # Setup optimizer with different learning rates if not provided
        if optimizer is None:
            self.optimizer = self._create_optimizer_with_different_lrs()
        else:
            self.optimizer = optimizer
        
        # Setup triangular LR schedule if scheduler not provided
        if scheduler is None:
            total_steps = max_steps if max_steps else len(train_dataloader) * max_epochs
            self.scheduler = self._create_triangular_lr_schedule(total_steps)
        else:
            self.scheduler = scheduler
    
    def _create_optimizer_with_different_lrs(self) -> AdamW:
        """
        Create optimizer with different learning rates for encoder and transformer.
        
        Encoder: low LR (encoder_lr_scale * peak_lr)
        Transformer: normal LR (peak_lr)
        """
        # Separate parameters into encoder and transformer groups
        encoder_params = []
        transformer_params = []
        
        # Identify encoder parameters (feature encoder from wav2vec2)
        if hasattr(self.model.model, 'feature_encoder'):
            encoder_params.extend(
                [p for p in self.model.model.feature_encoder.parameters() if p.requires_grad]
            )
        
        # All other parameters go to transformer group
        transformer_params.extend(
            [p for name, p in self.model.named_parameters() 
             if p.requires_grad and not name.startswith('model.feature_encoder')]
        )
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': encoder_params,
                'lr': self.peak_lr * self.encoder_lr_scale,
                'weight_decay': self.optimizer.weight_decay if hasattr(self, 'optimizer') else 0.01,
            },
            {
                'params': transformer_params,
                'lr': self.peak_lr,
                'weight_decay': self.optimizer.weight_decay if hasattr(self, 'optimizer') else 0.01,
            },
        ]
        
        logger.info(f"Optimizer groups:")
        logger.info(f"  Encoder params: {sum(p.numel() for p in encoder_params):,} (LR: {self.peak_lr * self.encoder_lr_scale:.2e})")
        logger.info(f"  Transformer params: {sum(p.numel() for p in transformer_params):,} (LR: {self.peak_lr:.2e})")
        
        return AdamW(
            param_groups,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01,
        )
    
    def _create_triangular_lr_schedule(self, total_steps: int):
        """
        Create triangular learning rate schedule with warmup.
        
        Schedule: linear warmup -> cosine annealing
        """
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,  # Start at 10% of peak LR
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )
        
        # Create cosine annealing scheduler (after warmup)
        cosine_steps = total_steps - self.warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=self.peak_lr * 0.01,  # Minimum LR is 1% of peak
        )
        
        # Combine into sequential scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )
        
        logger.info(f"Triangular LR schedule:")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Cosine steps: {cosine_steps}")
        logger.info(f"  Peak LR: {self.peak_lr:.2e}")
        logger.info(f"  Min LR: {self.peak_lr * 0.01:.2e}")
        
        return scheduler


def create_hybrid_model_and_config(args) -> tuple[HybridLaCTWav2Vec2Config, HybridLaCTWav2Vec2ForCTC]:
    """Create hybrid model and configuration."""
    
    # Load config from file if provided
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = HybridLaCTWav2Vec2Config(**config_dict)
    else:
        # Create default hybrid config
        config = HybridLaCTWav2Vec2Config(
            # Wav2Vec2 encoder
            use_wav2vec2_encoder=True,
            wav2vec2_model_name=args.wav2vec2_model_name or "facebook/wav2vec2-base-960h",
            encoder_target_ds_factor=5,  # 5x downsampling
            freeze_encoder=args.freeze_encoder,
            encoder_lr_scale=args.encoder_lr_scale or 0.1,
            
            # LaCT Transformer
            num_hidden_layers=16,  # 16 layers vs 12 in BASE
            hidden_size=768,
            intermediate_size=3072,  # FFN dimension
            num_attn_heads=8,
            num_lact_heads=4,
            lact_fw_hidden_dim=1536,
            
            # Fast-weight configuration
            fast_weight_lr_init=0.01,
            use_muon=True,
            use_momentum=True,
            
            # Dropout and regularization
            hidden_dropout=0.1,
            attention_dropout=0.1,
            layerdrop=0.05,
            
            # TTT configuration
            enable_ttt=args.enable_ttt,
            ttt_loss_type=args.ttt_loss_type or "masked_prediction",
            ttt_mask_prob=0.15,
            ttt_steps=1,
            
            # CTC
            ctc_vocab_size=args.vocab_size or 32,
            ctc_blank_id=0,
        )
    
    # Override config with command line arguments
    if args.vocab_size:
        config.ctc_vocab_size = args.vocab_size
        config.vocab_size = args.vocab_size
    
    # Create model
    model = HybridLaCTWav2Vec2ForCTC(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.model.feature_encoder.parameters()) if hasattr(model.model, 'feature_encoder') else 0
    encoder_trainable = sum(p.numel() for p in model.model.feature_encoder.parameters() if p.requires_grad) if hasattr(model.model, 'feature_encoder') else 0
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Encoder: {encoder_params:,} total ({encoder_trainable:,} trainable)")
    logger.info(f"  Transformer: {total_params - encoder_params:,} total ({trainable_params - encoder_trainable:,} trainable)")
    
    return config, model


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid LaCT + Wav2Vec2 ASR model")
    
    # Model arguments
    parser.add_argument("--config_path", type=str, help="Path to model config JSON file")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size (will be auto-detected if not provided)")
    parser.add_argument("--wav2vec2_model_name", type=str, default="facebook/wav2vec2-base-960h",
                       help="HuggingFace model name for wav2vec2 encoder")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze pretrained encoder")
    parser.add_argument("--encoder_lr_scale", type=float, default=0.1, help="Learning rate scale for encoder")
    
    # TTT arguments
    parser.add_argument("--enable_ttt", action="store_true", help="Enable Test-Time Training")
    parser.add_argument("--ttt_loss_type", type=str, choices=["masked_prediction", "entropy"],
                       default="masked_prediction", help="Type of TTT loss")
    
    # Data arguments (same as train_asr.py)
    parser.add_argument("--dataset_type", type=str, choices=["librispeech", "commonvoice", "generic"], 
                       default="librispeech", help="Type of dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--train_subset", type=str, default="train-clean-360", help="Training subset (for LibriSpeech)")
    parser.add_argument("--val_subset", type=str, default="dev-clean", help="Validation subset (for LibriSpeech)")
    parser.add_argument("--test_subset", type=str, help="Test subset (for LibriSpeech) - used for monitoring only")
    parser.add_argument("--test_eval_steps", type=int, default=5000, help="Steps between test set evaluations")
    parser.add_argument("--max_audio_duration", type=float, default=25.0, help="Maximum audio duration in seconds")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints/hybrid", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=25, help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps")
    parser.add_argument("--peak_lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm for clipping")
    parser.add_argument("--mixed_precision", action="store_true", 
                       help="Use mixed precision training")
    
    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    logger.info(f"Using device: {args.device}")
    logger.info("=" * 60)
    logger.info("Hybrid LaCT + Wav2Vec2 ASR Training")
    logger.info("=" * 60)
    
    # Get vocab size from dataset (same logic as train_asr.py)
    logger.info("Loading dataset to determine vocabulary size...")
    if args.dataset_type == "librispeech":
        from data import LibriSpeechDataset
        temp_dataset = LibriSpeechDataset(
            root_dir=args.data_dir,
            subset=args.train_subset,
            sample_rate=16000,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
    else:
        raise ValueError(f"Dataset type {args.dataset_type} not yet supported for hybrid model")
    
    actual_vocab_size = len(temp_dataset.vocab)
    logger.info(f"Dataset vocabulary size: {actual_vocab_size}")
    
    if not args.vocab_size:
        args.vocab_size = actual_vocab_size
    elif args.vocab_size != actual_vocab_size:
        logger.warning(f"Vocab size mismatch: args={args.vocab_size}, dataset={actual_vocab_size}")
        args.vocab_size = actual_vocab_size
    
    # Create model and config
    config, model = create_hybrid_model_and_config(args)
    
    # Create datasets and dataloaders (reuse from train_asr.py)
    train_dataloader, val_dataloader = create_datasets_and_dataloaders(args, config)
    
    # Create test dataloader if specified
    test_dataloader = None
    if args.test_subset and args.dataset_type == "librispeech":
        logger.info(f"Creating test dataloader for subset: {args.test_subset}")
        test_dataset = LibriSpeechDataset(
            root_dir=args.data_dir,
            subset=args.test_subset,
            sample_rate=config.sample_rate,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
        test_dataset.char_to_idx = temp_dataset.char_to_idx
        test_dataset.idx_to_char = temp_dataset.idx_to_char
        test_dataset.vocab = temp_dataset.vocab
        
        from data import ASRDataCollator, create_asr_dataloader
        collator = ASRDataCollator(hop_length=config.hop_length if hasattr(config, 'hop_length') else 160)
        test_dataloader = create_asr_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Create trainer
    trainer = HybridASRTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=args.device,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        test_eval_steps=args.test_eval_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        encoder_lr_scale=args.encoder_lr_scale,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
    )
    
    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Save config
    config_save_path = Path(args.output_dir) / "config.json"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved config to {config_save_path}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

