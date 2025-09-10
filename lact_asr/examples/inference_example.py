#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use LaCT ASR for inference.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.asr_inference import ASRInference


def main():
    # Example usage of LaCT ASR inference
    
    # Initialize inference (adjust paths as needed)
    model_path = "./checkpoints/best_model.pt"
    config_path = "./checkpoints/config.json"
    vocab_path = "./vocab.txt"  # Optional - will be created during training
    
    print("Initializing LaCT ASR model...")
    
    try:
        inference = ASRInference(
            model_path=model_path,
            config_path=config_path,
            vocab_path=vocab_path,
            device='cuda',  # or 'cpu'
            beam_width=1  # Greedy decoding
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained a model first using train_asr.py")
        return
    
    # Example 1: Transcribe a single audio file
    print("\n=== Single File Transcription ===")
    audio_file = "path/to/your/audio.wav"  # Replace with actual path
    
    try:
        result = inference.transcribe_file(audio_file)
        print(f"Audio file: {audio_file}")
        print(f"Transcription: {result['text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Audio duration: {result['audio_length']:.1f}s")
        
    except Exception as e:
        print(f"Error transcribing file: {e}")
        print("Please provide a valid audio file path")
    
    # Example 2: Batch transcription
    print("\n=== Batch Transcription ===")
    audio_files = [
        "path/to/audio1.wav",
        "path/to/audio2.wav",
        "path/to/audio3.wav"
    ]  # Replace with actual paths
    
    try:
        results = inference.transcribe_batch(audio_files, batch_size=4)
        
        for result in results:
            print(f"File: {result['audio_path']}")
            print(f"Transcription: {result['text']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error in batch transcription: {e}")
        print("Please provide valid audio file paths")
    
    print("\nInference examples completed!")


if __name__ == "__main__":
    main()
