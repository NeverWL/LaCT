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
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Test forward pass
        print(f"\nPerforming forward pass with real data...")
        
        with torch.no_grad():
            outputs = model(
                audio_input=batch['audio_input'],
                labels=batch['labels'],
                input_lengths=batch['input_lengths'],
                label_lengths=batch['label_lengths']
            )
        
        print(f"✓ Forward pass completed")
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  Logits range: [{outputs.logits.min():.4f}, {outputs.logits.max():.4f}]")
        print(f"  Logits mean: {outputs.logits.mean():.4f}")
        print(f"  Logits std: {outputs.logits.std():.4f}")
        
        # Check for NaN
        if torch.isnan(outputs.loss):
            print(f"\n❌ Loss is NaN with real data!")
            
            # Debug: check intermediate outputs
            print(f"\nDebugging intermediate outputs...")
            
            # Check hidden states
            if outputs.hidden_states is not None:
                for i, hidden in enumerate(outputs.hidden_states):
                    if torch.isnan(hidden).any():
                        print(f"  ❌ Layer {i} hidden states contain NaN")
                        break
                    else:
                        print(f"  ✓ Layer {i} hidden states OK - range [{hidden.min():.4f}, {hidden.max():.4f}]")
            
            return False
        
        if torch.isnan(outputs.logits).any():
            print(f"\n❌ Logits contain NaN with real data!")
            return False
        
        print(f"\n✅ Model works correctly with real data!")
        print(f"\nThis suggests the NaN issue occurs during:")
        print(f"  - Training dynamics (gradient updates)")
        print(f"  - Mixed precision training")
        print(f"  - Specific batches with unusual properties")
        
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

