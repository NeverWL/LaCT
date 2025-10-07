#!/usr/bin/env python3
"""
Alternative LibriSpeech download script using HuggingFace datasets.
This script can be used when openslr.org is not accessible due to firewall restrictions.

Usage:
    python download_librispeech_hf.py --data-dir /path/to/data --subsets train.clean.100 dev.clean
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import soundfile as sf
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install required packages:")
    print("  pip install datasets soundfile tqdm")
    sys.exit(1)


def download_librispeech_hf(data_dir: str, subsets: list):
    """
    Download LibriSpeech dataset from HuggingFace and save in openslr.org format.
    
    Args:
        data_dir: Directory to save the dataset
        subsets: List of subsets to download (e.g., ['train.clean.100', 'dev.clean'])
    """
    data_dir = Path(data_dir)
    librispeech_dir = data_dir / "LibriSpeech"
    
    print(f"Downloading LibriSpeech to: {librispeech_dir}")
    print(f"Subsets: {', '.join(subsets)}")
    print()
    
    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"Processing subset: {subset}")
        print(f"{'='*60}")
        
        # Convert subset name format (train-clean-100 -> train.clean.100)
        hf_subset = subset.replace('-', '.')
        
        try:
            # Load dataset from HuggingFace
            print(f"Loading {hf_subset} from HuggingFace...")
            dataset = load_dataset("librispeech_asr", hf_subset, split="train")
            
            # Create directory structure matching openslr.org format
            subset_dir = librispeech_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Converting {len(dataset)} samples to LibriSpeech format...")
            
            # Group samples by speaker and chapter
            samples_by_speaker = {}
            for sample in tqdm(dataset, desc=f"Processing {subset}"):
                speaker_id = sample['speaker_id']
                chapter_id = sample['chapter_id']
                utterance_id = sample['id']
                audio = sample['audio']
                text = sample['text']
                
                # Create speaker directory
                speaker_dir = subset_dir / str(speaker_id)
                speaker_dir.mkdir(exist_ok=True)
                
                # Create chapter directory
                chapter_dir = speaker_dir / str(chapter_id)
                chapter_dir.mkdir(exist_ok=True)
                
                # Save audio file
                audio_path = chapter_dir / f"{utterance_id}.flac"
                sf.write(str(audio_path), audio['array'], audio['sampling_rate'])
                
                # Collect transcription
                key = f"{speaker_id}/{chapter_id}"
                if key not in samples_by_speaker:
                    samples_by_speaker[key] = []
                samples_by_speaker[key].append((utterance_id, text))
            
            # Write transcription files
            print("Writing transcription files...")
            for key, samples in tqdm(samples_by_speaker.items(), desc="Writing transcripts"):
                speaker_id, chapter_id = key.split('/')
                chapter_dir = subset_dir / speaker_id / chapter_id
                trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                
                with open(trans_file, 'w') as f:
                    for utterance_id, text in sorted(samples):
                        f.write(f"{utterance_id} {text}\n")
            
            print(f"✓ Successfully downloaded {subset}")
            
        except Exception as e:
            print(f"✗ Error downloading {subset}: {e}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Download completed successfully!")
    print(f"{'='*60}")
    print(f"Dataset location: {librispeech_dir}")
    print("\nYou can now use this with the training script:")
    print(f"  ./examples/train_librispeech.sh --data-dir {librispeech_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download LibriSpeech dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download minimal dataset
  python %(prog)s -d /data/LibriSpeech -s train.clean.100 dev.clean
  
  # Download standard dataset
  python %(prog)s -d /data/LibriSpeech -s train.clean.360 dev.clean
  
Available subsets:
  train.clean.100, train.clean.360, train.other.500
  dev.clean, dev.other, test.clean, test.other
        """
    )
    
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        required=True,
        help='Directory to save LibriSpeech dataset'
    )
    
    parser.add_argument(
        '-s', '--subsets',
        type=str,
        nargs='+',
        required=True,
        help='Subsets to download (space-separated)'
    )
    
    args = parser.parse_args()
    
    # Convert subset names if needed (train-clean-100 -> train.clean.100)
    subsets = [s.replace('-', '.') if '-' in s else s for s in args.subsets]
    
    download_librispeech_hf(args.data_dir, subsets)


if __name__ == "__main__":
    main()

