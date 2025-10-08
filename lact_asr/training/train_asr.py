#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from transformers import get_linear_schedule_with_warmup

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lact_asr_model import (
    LaCTASRConfig, 
    LaCTASRForCTC, 
    AudioFeatureExtractor,
    MelSpectrogramExtractor
)
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


class ASRTrainer:
    """
    Trainer class for LaCT ASR models.
    """
    
    def __init__(
        self,
        config: LaCTASRConfig,
        model: LaCTASRForCTC,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './checkpoints',
        save_steps: int = 1000,
        eval_steps: int = 500,
        logging_steps: int = 100,
        max_steps: Optional[int] = None,
        max_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
    ):
        self.config = config
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        
        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=5e-6,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # Setup scheduler
        if scheduler is None:
            total_steps = max_steps if max_steps else len(train_dataloader) * max_epochs
            warmup_steps = total_steps // 10
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = scheduler
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.max_epochs}")
        logger.info(f"Total steps: {self.max_steps or 'unlimited'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        
        self.model.train()
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.max_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Detailed logging for first 10 steps
                if self.global_step < 10:
                    logger.info(f"=== Batch {batch_idx} Details (Step {self.global_step}) ===")
                    logger.info(f"  Audio input shape: {batch['audio_input'].shape}")
                    logger.info(f"  Audio input range: [{batch['audio_input'].min():.4f}, {batch['audio_input'].max():.4f}]")
                    logger.info(f"  Input lengths: {batch['input_lengths']}")
                    logger.info(f"  Label lengths: {batch['label_lengths']}")
                    logger.info(f"  Labels shape: {batch['labels'].shape}")
                
                # Forward pass
                loss = self._training_step(batch)
                
                # Skip batch if validation failed
                if loss is None:
                    logger.warning(f"Skipping batch {batch_idx} due to validation failure")
                    continue
                
                # Log loss for first 10 steps
                if self.global_step < 10:
                    logger.info(f"  Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                
                # Backward pass
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                epoch_loss += loss.item()
                num_batches += 1
                
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"Step {self.global_step} | Epoch {epoch + 1} | "
                        f"Batch {batch_idx + 1}/{len(self.train_dataloader)} | "
                        f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                    )
                
                # Validation
                if self.val_dataloader and self.global_step % self.eval_steps == 0:
                    val_loss = self._validate()
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(is_best=True)
                        logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint()
                
                # Check max steps
                if self.max_steps and self.global_step >= self.max_steps:
                    logger.info(f"Reached maximum steps: {self.max_steps}")
                    return
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.train_losses.append(avg_epoch_loss)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint at end of epoch
            self._save_checkpoint()
        
        logger.info("Training completed!")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        # Comprehensive input validation
        # Return None to signal batch should be skipped
        if torch.isnan(batch['audio_input']).any() or torch.isinf(batch['audio_input']).any():
            logger.error("NaN or Inf detected in audio_input! Skipping batch.")
            return None
        
        # Check for invalid CTC inputs
        if (batch['input_lengths'] <= 0).any():
            logger.error(f"Invalid input_lengths detected: {batch['input_lengths']}")
            return None
        
        if (batch['label_lengths'] <= 0).any():
            logger.error(f"Invalid label_lengths detected: {batch['label_lengths']}")
            return None
        
        # CTC requires input_length >= label_length
        if (batch['input_lengths'] < batch['label_lengths']).any():
            logger.error(f"CTC constraint violated: input_lengths < label_lengths")
            logger.error(f"  input_lengths: {batch['input_lengths']}")
            logger.error(f"  label_lengths: {batch['label_lengths']}")
            return None
        
        # Log forward pass for first few steps
        if self.global_step < 10:
            logger.info(f"  Starting forward pass...")
        
        if self.mixed_precision:
            # Use mixed precision but with specific dtype control
            # Some operations need to stay in float32 to avoid NaN
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(
                    audio_input=batch['audio_input'],
                    labels=batch['labels'],
                    label_lengths=batch['label_lengths'],
                    input_lengths=batch['input_lengths']
                )
                loss = outputs.loss
        else:
            outputs = self.model(
                audio_input=batch['audio_input'],
                labels=batch['labels'],
                label_lengths=batch['label_lengths'],
                input_lengths=batch['input_lengths']
            )
            loss = outputs.loss
        
        # Log after forward pass for first few steps
        if self.global_step < 10:
            logger.info(f"  Forward pass complete. Loss: {loss.item() if torch.isfinite(loss) else 'NaN/Inf'}")
            logger.info(f"  Logits shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")
            if hasattr(outputs, 'logits'):
                logger.info(f"  Logits range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
        
        # Check for NaN/Inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN or Inf loss detected at step {self.global_step}!")
            logger.error(f"  Input lengths: {batch['input_lengths']}")
            logger.error(f"  Label lengths: {batch['label_lengths']}")
            logger.error(f"  Audio input shape: {batch['audio_input'].shape}")
            logger.error(f"  Audio input stats - min: {batch['audio_input'].min():.4f}, "
                        f"max: {batch['audio_input'].max():.4f}, "
                        f"mean: {batch['audio_input'].mean():.4f}")
            
            # Check if model weights have NaN
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    logger.error(f"  NaN/Inf in model parameter: {name}")
            
            # Log which parameters have NaN gradients (but don't iterate if no gradients yet)
            if self.global_step > 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        logger.error(f"  NaN/Inf gradient in parameter: {name}")
            
            # Return None to skip this batch
            logger.warning(f"Skipping batch due to NaN/Inf loss")
            return None
        
        return loss
    
    def _validate(self) -> float:
        """Run validation loop."""
        if not self.val_dataloader:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            audio_input=batch['audio_input'],
                            labels=batch['labels'],
                            label_lengths=batch['label_lengths'],
                            input_lengths=batch['input_lengths']
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        audio_input=batch['audio_input'],
                        labels=batch['labels'],
                        label_lengths=batch['label_lengths'],
                        input_lengths=batch['input_lengths']
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint-step-{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        # Save latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from step {self.global_step}")


def create_model_and_config(args) -> tuple[LaCTASRConfig, LaCTASRForCTC]:
    """Create model and configuration."""
    
    # Load config from file if provided
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config_dict = json.load(f)
        config = LaCTASRConfig(**config_dict)
    else:
        # Create default config
        config = LaCTASRConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attn_heads=12,
            num_lact_heads=4,
            lact_chunk_size=4096,
            window_size=4096,
            max_position_embeddings=16384,
            ctc_vocab_size=args.vocab_size,
            audio_feature_dim=80,
            sample_rate=16000,
            use_muon=True,
            use_momentum=True,
            learnable_ttt_scale=True,
        )
    
    # Override config with command line arguments
    if args.vocab_size:
        config.ctc_vocab_size = args.vocab_size
        config.vocab_size = args.vocab_size
    
    # Create model
    model = LaCTASRForCTC(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return config, model


def create_datasets_and_dataloaders(args, config: LaCTASRConfig) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create datasets and dataloaders."""
    
    # Create datasets based on dataset type
    if args.dataset_type == "librispeech":
        train_dataset = LibriSpeechDataset(
            root_dir=args.data_dir,
            subset=args.train_subset or "train-clean-100",
            sample_rate=config.sample_rate,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
        
        if args.val_subset:
            val_dataset = LibriSpeechDataset(
                root_dir=args.data_dir,
                subset=args.val_subset,
                sample_rate=config.sample_rate,
                max_duration=args.max_audio_duration,
                normalize_text=True,
                vocab_path=None  # Use same vocab as train
            )
            # Share vocabulary
            val_dataset.vocab = train_dataset.vocab
            val_dataset.char_to_idx = train_dataset.char_to_idx
            val_dataset.idx_to_char = train_dataset.idx_to_char
        else:
            val_dataset = None
            
    elif args.dataset_type == "commonvoice":
        train_dataset = CommonVoiceDataset(
            root_dir=args.data_dir,
            split="train",
            language=args.language or "en",
            sample_rate=config.sample_rate,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
        
        val_dataset = CommonVoiceDataset(
            root_dir=args.data_dir,
            split="dev",
            language=args.language or "en",
            sample_rate=config.sample_rate,
            max_duration=args.max_audio_duration,
            normalize_text=True,
            vocab_path=None
        )
        # Share vocabulary
        val_dataset.vocab = train_dataset.vocab
        val_dataset.char_to_idx = train_dataset.char_to_idx
        val_dataset.idx_to_char = train_dataset.idx_to_char
        
    else:  # generic dataset
        train_dataset = ASRDataset(
            manifest_path=args.train_manifest,
            vocab_path=args.vocab_path,
            sample_rate=config.sample_rate,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
        
        val_dataset = None
        if args.val_manifest:
            val_dataset = ASRDataset(
                manifest_path=args.val_manifest,
                vocab_path=args.vocab_path,
                sample_rate=config.sample_rate,
                max_duration=args.max_audio_duration,
                normalize_text=True,
            )
    
    # Update config with actual vocab size
    config.ctc_vocab_size = len(train_dataset.vocab)
    config.vocab_size = len(train_dataset.vocab)
    
    # Create data collator
    collator = ASRDataCollator(
        pad_token_id=0,  # blank token
        max_audio_length=None,
        max_text_length=None,
    )
    
    # Create dataloaders
    train_dataloader = create_asr_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_asr_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Vocabulary size: {len(train_dataset.vocab)}")
    
    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description="Train LaCT ASR model")
    
    # Model arguments
    parser.add_argument("--config_path", type=str, help="Path to model config JSON file")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size (will be auto-detected if not provided)")
    
    # Data arguments
    parser.add_argument("--dataset_type", type=str, choices=["librispeech", "commonvoice", "generic"], 
                       default="generic", help="Type of dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--train_manifest", type=str, help="Path to training manifest (for generic dataset)")
    parser.add_argument("--val_manifest", type=str, help="Path to validation manifest (for generic dataset)")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    parser.add_argument("--train_subset", type=str, help="Training subset (for LibriSpeech)")
    parser.add_argument("--val_subset", type=str, help="Validation subset (for LibriSpeech)")
    parser.add_argument("--language", type=str, default="en", help="Language code (for Common Voice)")
    parser.add_argument("--max_audio_duration", type=float, default=20.0, help="Maximum audio duration in seconds")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--mixed_precision", action="store_true", 
                        help="Use mixed precision training (WARNING: May cause NaN, disabled by default)")
    
    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps")
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
    
    # First, create a temporary dataset to get vocab size
    logger.info("Loading dataset to determine vocabulary size...")
    if args.dataset_type == "librispeech":
        temp_dataset = LibriSpeechDataset(
            root_dir=args.data_dir,
            subset=args.train_subset or "train-clean-100",
            sample_rate=16000,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
    elif args.dataset_type == "commonvoice":
        temp_dataset = CommonVoiceDataset(
            root_dir=args.data_dir,
            split="train",
            language=args.language or "en",
            sample_rate=16000,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
    else:
        temp_dataset = ASRDataset(
            manifest_path=args.train_manifest,
            vocab_path=args.vocab_path,
            sample_rate=16000,
            max_duration=args.max_audio_duration,
            normalize_text=True,
        )
    
    # Get actual vocab size from dataset
    actual_vocab_size = len(temp_dataset.vocab)
    logger.info(f"Dataset vocabulary size: {actual_vocab_size}")
    logger.info(f"Vocabulary: {temp_dataset.vocab}")
    
    # Override vocab_size argument with actual vocab size
    if not args.vocab_size:
        args.vocab_size = actual_vocab_size
        logger.info(f"Setting vocab_size to {actual_vocab_size}")
    elif args.vocab_size != actual_vocab_size:
        logger.warning(f"Vocab size mismatch: args={args.vocab_size}, dataset={actual_vocab_size}")
        logger.warning(f"Using dataset vocab size: {actual_vocab_size}")
        args.vocab_size = actual_vocab_size
    
    # Create model and config with correct vocab size
    config, model = create_model_and_config(args)
    
    # Create datasets and dataloaders (will reuse the vocab from temp_dataset)
    train_dataloader, val_dataloader = create_datasets_and_dataloaders(args, config)
    
    # Verify vocab size matches
    logger.info(f"Model CTC head vocab size: {model.ctc_head.out_features}")
    logger.info(f"Config vocab size: {config.ctc_vocab_size}")
    
    if model.ctc_head.out_features != actual_vocab_size:
        logger.error(f"CRITICAL: Vocab size mismatch!")
        logger.error(f"  Model CTC head: {model.ctc_head.out_features}")
        logger.error(f"  Dataset vocab: {actual_vocab_size}")
        raise ValueError(f"Vocab size mismatch: model has {model.ctc_head.out_features}, dataset has {actual_vocab_size}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = ASRTrainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=args.device,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
    )
    
    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # Save config
    config_save_path = Path(args.output_dir) / "config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Saved config to {config_save_path}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
