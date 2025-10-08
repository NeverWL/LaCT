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

if __name__ == "__main__":
    success = check_model_initialization()
    sys.exit(0 if success else 1)

