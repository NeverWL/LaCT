#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify LaCT ASR setup and basic functionality.
"""

import torch
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")
    
    try:
        from lact_asr_model import (
            LaCTASRConfig, 
            LaCTASRModel, 
            LaCTASRForCTC,
            AudioFeatureExtractor,
            MelSpectrogramExtractor,
            LaCTASRLayer
        )
        print("✓ Model components imported successfully")
    except Exception as e:
        print(f"✗ Error importing model components: {e}")
        return False
    
    try:
        from data import (
            ASRDataset,
            LibriSpeechDataset,
            CommonVoiceDataset,
            ASRDataCollator,
            create_vocab_from_transcripts
        )
        print("✓ Data components imported successfully")
    except Exception as e:
        print(f"✗ Error importing data components: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and basic forward pass."""
    print("\nTesting model creation...")
    
    try:
        from lact_asr_model import LaCTASRConfig, LaCTASRForCTC
        
        # Create config
        config = LaCTASRConfig(
            hidden_size=256,  # Small for testing
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            ctc_vocab_size=32,
            audio_feature_dim=80,
            sample_rate=16000,
            lact_chunk_size=1024,  # Small for testing
            window_size=1024,
            max_position_embeddings=2048
        )
        
        # Create model
        model = LaCTASRForCTC(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size = 2
        seq_len = 1000  # 1000 audio samples
        
        # Create dummy audio input
        audio_input = torch.randn(batch_size, seq_len)
        
        # Create dummy labels for CTC
        labels = torch.randint(1, config.ctc_vocab_size, (batch_size, 10))  # 10 character sequence
        label_lengths = torch.tensor([8, 10])
        input_lengths = torch.tensor([seq_len // 160, seq_len // 160])  # Approximate frame count
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                audio_input=audio_input,
                labels=labels,
                label_lengths=label_lengths,
                input_lengths=input_lengths
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - Output logits shape: {outputs.logits.shape}")
        print(f"  - Loss: {outputs.loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model creation/forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_features():
    """Test audio feature extraction."""
    print("\nTesting audio feature extraction...")
    
    try:
        from lact_asr_model import AudioFeatureExtractor, LaCTASRConfig
        
        config = LaCTASRConfig()
        feature_extractor = AudioFeatureExtractor(config, input_type="mel")
        
        # Create dummy audio
        batch_size = 2
        audio_length = 16000  # 1 second at 16kHz
        audio_input = torch.randn(batch_size, audio_length)
        
        # Extract features
        features = feature_extractor(audio_input)
        
        print(f"✓ Audio feature extraction successful")
        print(f"  - Input shape: {audio_input.shape}")
        print(f"  - Output features shape: {features.shape}")
        print(f"  - Feature dimension: {features.shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in audio feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_components():
    """Test data loading components."""
    print("\nTesting data components...")
    
    try:
        from data import ASRDataCollator
        
        # Test data collator
        collator = ASRDataCollator(pad_token_id=0)
        
        # Create dummy batch
        batch = [
            {
                'audio': torch.randn(1600),  # 0.1 seconds
                'audio_length': 1600,
                'text': 'hello world',
                'text_indices': torch.tensor([1, 2, 3, 4, 5]),
                'text_length': 5,
                'duration': 0.1,
                'audio_filepath': 'dummy1.wav'
            },
            {
                'audio': torch.randn(3200),  # 0.2 seconds
                'audio_length': 3200,
                'text': 'test',
                'text_indices': torch.tensor([1, 2, 3]),
                'text_length': 3,
                'duration': 0.2,
                'audio_filepath': 'dummy2.wav'
            }
        ]
        
        # Collate batch
        collated = collator(batch)
        
        print(f"✓ Data collation successful")
        print(f"  - Audio batch shape: {collated['audio_input'].shape}")
        print(f"  - Label batch shape: {collated['labels'].shape}")
        print(f"  - Input lengths: {collated['input_lengths']}")
        print(f"  - Label lengths: {collated['label_lengths']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data components: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("LaCT ASR Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_passed = False
    
    # Test audio features
    if not test_audio_features():
        all_passed = False
    
    # Test data components
    if not test_data_components():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed! LaCT ASR setup is working correctly.")
        print("\nNext steps:")
        print("1. Prepare your audio dataset in manifest format")
        print("2. Run training: python training/train_asr.py --help")
        print("3. Use inference: python inference/asr_inference.py --help")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Install flash-linear-attention: pip install flash-linear-attention")
        print("3. Check that the lact_llm directory is accessible")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
