#!/usr/bin/env python3
"""
Diagnostic script to check if model initializes correctly and can perform a forward pass.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lact_asr_model import LaCTASRConfig, LaCTASRForCTC

def check_model_initialization():
    """Check if model initializes without NaN."""
    print("=" * 60)
    print("Model Initialization Diagnostic")
    print("=" * 60)
    
    # Create config
    config = LaCTASRConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attn_heads=12,
        num_lact_heads=4,
        ctc_vocab_size=32,
    )
    
    print(f"\n✓ Config created")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Vocab size: {config.ctc_vocab_size}")
    
    # Create model
    print(f"\nInitializing model...")
    model = LaCTASRForCTC(config)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    device = next(model.parameters()).device
    print(f"✓ Model created on device: {device}")
    
    # Check for NaN in initial weights
    print(f"\nChecking model parameters...")
    nan_params = []
    inf_params = []
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
    
    print(f"✓ Total parameters: {total_params}")
    
    if nan_params:
        print(f"\n❌ Found NaN in {len(nan_params)} parameters:")
        for name in nan_params[:10]:  # Show first 10
            print(f"     - {name}")
        return False
    
    if inf_params:
        print(f"\n❌ Found Inf in {len(inf_params)} parameters:")
        for name in inf_params[:10]:
            print(f"     - {name}")
        return False
    
    print(f"✓ No NaN/Inf in model parameters")
    
    # Create dummy input
    print(f"\nCreating dummy input...")
    batch_size = 2
    audio_length = 16000  # 1 second of audio at 16kHz
    
    dummy_audio = torch.randn(batch_size, audio_length, device=device) * 0.1
    dummy_labels = torch.randint(1, 32, (batch_size, 20), device=device)
    input_lengths = torch.tensor([100, 100], device=device)  # After mel-spec: 16000/160=100
    label_lengths = torch.tensor([20, 20], device=device)
    
    print(f"✓ Dummy input created")
    print(f"  Audio shape: {dummy_audio.shape}")
    print(f"  Labels shape: {dummy_labels.shape}")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Label lengths: {label_lengths}")
    
    # Test forward pass
    print(f"\nPerforming forward pass...")
    model.eval()
    
    try:
        with torch.no_grad():
            outputs = model(
                audio_input=dummy_audio,
                labels=dummy_labels,
                input_lengths=input_lengths,
                label_lengths=label_lengths
            )
        
        print(f"✓ Forward pass completed")
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  Logits range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
        
        # Check outputs for NaN
        if torch.isnan(outputs.loss):
            print(f"\n❌ Loss is NaN!")
            return False
        
        if torch.isnan(outputs.logits).any():
            print(f"\n❌ Logits contain NaN!")
            return False
        
        print(f"\n✅ Model appears healthy!")
        return True
        
    except Exception as e:
        print(f"\n❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data(data_dir: str, subset: str = "train-clean-100"):
    """Test model with actual LibriSpeech data."""
    print("\n" + "=" * 60)
    print("Testing with Real LibriSpeech Data")
    print("=" * 60)
    
    # Import data modules
    from data import LibriSpeechDataset, ASRDataCollator
    from torch.utils.data import DataLoader
    
    # Create config and model
    config = LaCTASRConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attn_heads=12,
        num_lact_heads=4,
        ctc_vocab_size=32,
    )
    
    print(f"\nLoading dataset from: {data_dir}")
    print(f"Subset: {subset}")
    
    try:
        # Create dataset
        dataset = LibriSpeechDataset(
            root_dir=data_dir,
            subset=subset,
            max_duration=20.0,
            min_duration=0.5
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"  Vocabulary size: {len(dataset.vocab)}")
        print(f"  Vocabulary: {dataset.vocab}")
        print(f"\n  Character mapping:")
        for idx, char in enumerate(dataset.vocab[:10]):  # Show first 10
            print(f"    {idx}: '{char}'")
        if len(dataset.vocab) > 10:
            print(f"    ... ({len(dataset.vocab) - 10} more)")
        
        # Update config with actual vocab size
        print(f"\nUpdating config vocab size from {config.ctc_vocab_size} to {len(dataset.vocab)}")
        config.ctc_vocab_size = len(dataset.vocab)
        config.vocab_size = len(dataset.vocab)
        
        # Create dataloader with just 1 batch
        collator = ASRDataCollator(hop_length=config.hop_length)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collator,
            num_workers=0  # No workers for debugging
        )
        
        # Get first batch
        print(f"\nGetting first batch from dataset...")
        batch = next(iter(dataloader))
        
        print(f"✓ Batch loaded")
        print(f"  Audio input shape: {batch['audio_input'].shape}")
        print(f"  Audio input range: [{batch['audio_input'].min():.4f}, {batch['audio_input'].max():.4f}]")
        print(f"  Audio input mean: {batch['audio_input'].mean():.4f}")
        print(f"  Audio input std: {batch['audio_input'].std():.4f}")
        print(f"  Input lengths: {batch['input_lengths']}")
        print(f"  Label lengths: {batch['label_lengths']}")
        print(f"  Labels shape: {batch['labels'].shape}")
        print(f"  Labels min/max: [{batch['labels'].min().item()}, {batch['labels'].max().item()}]")
        print(f"  Unique labels in batch: {torch.unique(batch['labels']).tolist()}")
        
        # Check if labels exceed vocab size
        if batch['labels'].max() >= len(dataset.vocab):
            print(f"\n❌ Labels exceed vocabulary size!")
            print(f"   Max label: {batch['labels'].max().item()}")
            print(f"   Vocab size: {len(dataset.vocab)}")
            print(f"   This will cause indexing errors!")
            return False
        
        # Check for NaN/Inf in batch
        if torch.isnan(batch['audio_input']).any():
            print(f"\n❌ Audio input contains NaN!")
            return False
        
        if torch.isinf(batch['audio_input']).any():
            print(f"\n❌ Audio input contains Inf!")
            return False
        
        # Create model
        print(f"\nCreating model...")
        model = LaCTASRForCTC(config)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        print(f"✓ Model created")
        print(f"  CTC head vocab size: {model.ctc_head.out_features}")
        print(f"  Dataset vocab size: {len(dataset.vocab)}")
        
        if model.ctc_head.out_features != len(dataset.vocab):
            print(f"\n⚠️  WARNING: Vocab size mismatch!")
            print(f"     CTC head: {model.ctc_head.out_features}")
            print(f"     Dataset:  {len(dataset.vocab)}")
            print(f"     This could cause issues if labels >= {model.ctc_head.out_features}")
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Test forward pass WITHOUT labels first (no loss computation)
        print(f"\nPerforming forward pass with real data (no loss)...")
        
        with torch.no_grad():
            # Forward through model without computing loss
            outputs_no_loss = model.model(
                audio_input=batch['audio_input']
            )
        
        hidden_states = outputs_no_loss[0]
        
        print(f"✓ Model forward pass completed")
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Hidden states range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
        print(f"  Hidden states mean: {hidden_states.mean():.4f}")
        print(f"  Hidden states std: {hidden_states.std():.4f}")
        
        # Check hidden states for NaN
        if torch.isnan(hidden_states).any():
            print(f"\n❌ Hidden states contain NaN!")
            print(f"   NaN appeared in the model layers (before CTC head)")
            return False
        
        # Apply CTC head manually
        print(f"\nApplying CTC head...")
        logits = model.ctc_head(hidden_states)
        
        print(f"✓ CTC head applied")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  Logits mean: {logits.mean():.4f}")
        print(f"  Logits std: {logits.std():.4f}")
        
        # Check logits for NaN
        if torch.isnan(logits).any():
            print(f"\n❌ Logits contain NaN!")
            print(f"   NaN appeared in CTC head projection")
            return False
        
        # Apply log softmax
        print(f"\nApplying log_softmax...")
        log_probs = torch.log_softmax(logits, dim=-1)
        
        print(f"✓ Log softmax applied")
        print(f"  Log probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
        print(f"  Log probs mean: {log_probs.mean():.4f}")
        
        # Check log_probs for NaN
        if torch.isnan(log_probs).any():
            print(f"\n❌ Log probs contain NaN!")
            print(f"   NaN appeared in log_softmax (likely from -inf in logits)")
            return False
        
        # Prepare for CTC loss
        print(f"\nPreparing CTC loss inputs...")
        
        labels = batch['labels']
        input_lengths = batch['input_lengths']
        label_lengths = batch['label_lengths']
        
        # Validate CTC constraints
        print(f"  Input lengths: {input_lengths.tolist()}")
        print(f"  Label lengths: {label_lengths.tolist()}")
        print(f"  Max input length: {input_lengths.max().item()}")
        print(f"  Sequence length: {hidden_states.size(1)}")
        
        if (input_lengths > hidden_states.size(1)).any():
            print(f"\n❌ input_lengths exceed sequence length!")
            print(f"   Some input_lengths are larger than {hidden_states.size(1)}")
            return False
        
        if (input_lengths < label_lengths).any():
            print(f"\n❌ CTC constraint violated: input_lengths < label_lengths")
            return False
        
        print(f"✓ CTC constraints satisfied")
        
        # Transpose for CTC loss: [batch, time, vocab] -> [time, batch, vocab]
        print(f"\nTransposing for CTC loss...")
        log_probs_transposed = log_probs.transpose(0, 1)
        print(f"  Transposed shape: {log_probs_transposed.shape}")
        
        # Compute CTC loss
        print(f"\nComputing CTC loss...")
        ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        loss = ctc_loss_fn(
            log_probs_transposed,
            labels,
            input_lengths,
            label_lengths
        )
        
        print(f"✓ CTC loss computed")
        print(f"  Loss value: {loss.item():.4f}")
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"\n❌ Loss is NaN!")
            print(f"   NaN appeared during CTC loss computation")
            print(f"\nDebugging CTC inputs:")
            print(f"  log_probs contains -inf: {torch.isinf(log_probs).any().item()}")
            print(f"  log_probs min: {log_probs.min().item():.4f}")
            print(f"  log_probs max: {log_probs.max().item():.4f}")
            print(f"  labels min: {labels.min().item()}")
            print(f"  labels max: {labels.max().item()}")
            return False
        
        if torch.isinf(loss):
            print(f"\n❌ Loss is Inf!")
            return False
        
        print(f"\n✅ Model works correctly with real data in eval mode!")
        
        # Now test in TRAINING mode WITHOUT mixed precision first
        print(f"\n" + "=" * 60)
        print("Testing in TRAINING MODE (FP32, no mixed precision)")
        print("=" * 60)
        
        model.train()  # Switch to training mode
        
        print(f"\nPerforming forward pass in training mode (FP32)...")
        
        # Test WITHOUT mixed precision first
        outputs_train_fp32 = model.model(
            audio_input=batch['audio_input']
        )
        
        hidden_states_train_fp32 = outputs_train_fp32[0]
        
        print(f"✓ Training mode forward pass completed (FP32)")
        print(f"  Hidden states shape: {hidden_states_train_fp32.shape}")
        print(f"  Hidden states range: [{hidden_states_train_fp32.min():.4f}, {hidden_states_train_fp32.max():.4f}]")
        
        if torch.isnan(hidden_states_train_fp32).any():
            print(f"\n❌ Hidden states contain NaN in training mode (even without mixed precision)!")
            print(f"   The issue is in training mode itself, not mixed precision")
            print(f"\nLikely causes:")
            print(f"  - Dropout causing instability")
            print(f"  - BatchNorm in training mode")
            print(f"  - TTT layer numerical issues")
            
            # Test with dropout disabled
            print(f"\nTesting with dropout disabled...")
            model.eval()  # This disables dropout
            
            # But manually enable BatchNorm training mode
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.train()
            
            outputs_no_dropout = model.model(audio_input=batch['audio_input'])
            hidden_no_dropout = outputs_no_dropout[0]
            
            if torch.isnan(hidden_no_dropout).any():
                print(f"  ❌ Still NaN with dropout disabled - issue is in BatchNorm or core layers")
            else:
                print(f"  ✓ Works without dropout - Dropout is causing NaN!")
                print(f"     Hidden states range: [{hidden_no_dropout.min():.4f}, {hidden_no_dropout.max():.4f}]")
            
            return False
        
        print(f"✓ Training mode works in FP32!")
        
        # Now test with mixed precision
        print(f"\n" + "=" * 60)
        print("Testing in TRAINING MODE with Mixed Precision (FP16)")
        print("=" * 60)
        
        model.train()
        
        print(f"\nPerforming forward pass in training mode with AMP...")
        
        # Use mixed precision like actual training
        with torch.cuda.amp.autocast():
            outputs_train = model.model(
                audio_input=batch['audio_input']
            )
        
        hidden_states_train = outputs_train[0]
        
        print(f"✓ Training mode forward pass completed (FP16)")
        print(f"  Hidden states shape: {hidden_states_train.shape}")
        print(f"  Hidden states range: [{hidden_states_train.min():.4f}, {hidden_states_train.max():.4f}]")
        
        if torch.isnan(hidden_states_train).any():
            print(f"\n❌ Hidden states contain NaN with mixed precision!")
            print(f"   Mixed precision (FP16) causes NaN")
            print(f"   FP32 works fine, so use training without --mixed_precision flag")
            return False
        
        # Apply CTC head
        logits_train = model.ctc_head(hidden_states_train)
        
        print(f"✓ CTC head applied in training mode")
        print(f"  Logits range: [{logits_train.min():.4f}, {logits_train.max():.4f}]")
        
        if torch.isnan(logits_train).any():
            print(f"\n❌ Logits contain NaN in training mode!")
            return False
        
        # Compute loss with mixed precision
        log_probs_train = torch.log_softmax(logits_train, dim=-1)
        log_probs_transposed_train = log_probs_train.transpose(0, 1)
        
        ctc_loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        loss_train = ctc_loss_fn(
            log_probs_transposed_train,
            labels,
            input_lengths,
            label_lengths
        )
        
        print(f"✓ CTC loss computed in training mode")
        print(f"  Loss value: {loss_train.item():.4f}")
        
        if torch.isnan(loss_train):
            print(f"\n❌ Loss is NaN in training mode with mixed precision!")
            print(f"   This confirms mixed precision is causing the issue")
            return False
        
        # Test backward pass
        print(f"\nTesting backward pass...")
        loss_train.backward()
        
        print(f"✓ Backward pass completed")
        
        # Check gradients for NaN
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                nan_grads.append(name)
        
        if nan_grads:
            print(f"\n❌ Found NaN/Inf in gradients:")
            for name in nan_grads[:10]:
                print(f"     - {name}")
            return False
        
        print(f"✓ No NaN/Inf in gradients")
        
        print(f"\n✅ First batch works in training mode with mixed precision!")
        
        # Test multiple batches to find problematic ones
        print(f"\n" + "=" * 60)
        print("Testing Multiple Random Batches")
        print("=" * 60)
        
        # Create a shuffled dataloader like training
        dataloader_shuffled = DataLoader(
            dataset,
            batch_size=8,  # Same as training
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )
        
        print(f"\nTesting 10 random batches...")
        model.train()
        
        for batch_idx, test_batch in enumerate(dataloader_shuffled):
            if batch_idx >= 10:
                break
            
            # Move to device
            test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in test_batch.items()}
            
            # Quick forward pass
            try:
                with torch.cuda.amp.autocast():
                    test_outputs = model(
                        audio_input=test_batch['audio_input'],
                        labels=test_batch['labels'],
                        input_lengths=test_batch['input_lengths'],
                        label_lengths=test_batch['label_lengths']
                    )
                
                if torch.isnan(test_outputs.loss) or torch.isnan(test_outputs.logits).any():
                    print(f"\n❌ Batch {batch_idx} produces NaN!")
                    print(f"   Audio shape: {test_batch['audio_input'].shape}")
                    print(f"   Input lengths: {test_batch['input_lengths'].tolist()}")
                    print(f"   Label lengths: {test_batch['label_lengths'].tolist()}")
                    print(f"   Labels min/max: [{test_batch['labels'].min()}, {test_batch['labels'].max()}]")
                    return False
                else:
                    print(f"  ✓ Batch {batch_idx}: Loss={test_outputs.loss.item():.4f}, "
                          f"Logits range=[{test_outputs.logits.min():.4f}, {test_outputs.logits.max():.4f}]")
            
            except Exception as e:
                print(f"\n❌ Batch {batch_idx} failed: {e}")
                return False
        
        print(f"\n✅ All 10 random batches processed successfully!")
        print(f"\nIf training still gets NaN, the issue is likely:")
        print(f"  - Accumulation of numerical errors over many steps")
        print(f"  - Optimizer updates causing parameter corruption")
        print(f"  - Learning rate schedule causing instability after warmup")
        print(f"  - GradScaler state issues in mixed precision")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing with real data:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose LaCT ASR model")
    parser.add_argument("--data-dir", type=str, help="Path to LibriSpeech dataset for real data testing")
    parser.add_argument("--subset", type=str, default="train-clean-100", help="Dataset subset to test")
    
    args = parser.parse_args()
    
    # Test with dummy data
    success = check_model_initialization()
    
    # Test with real data if provided
    if args.data_dir and success:
        success = test_with_real_data(args.data_dir, args.subset)
    
    sys.exit(0 if success else 1)

