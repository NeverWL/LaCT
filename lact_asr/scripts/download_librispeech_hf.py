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
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("\nPlease install required packages:")
    print("  pip install datasets soundfile tqdm numpy")
    sys.exit(1)

# Optional: matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend for HPC
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def validate_dataset(librispeech_dir: Path, subsets: list) -> bool:
    """
    Validate the downloaded LibriSpeech dataset structure and content.
    
    Args:
        librispeech_dir: Path to LibriSpeech directory
        subsets: List of subsets that should be present
    
    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n{'='*60}")
    print("Validating Dataset Structure")
    print(f"{'='*60}")
    
    all_valid = True
    total_audio_files = 0
    total_transcript_files = 0
    
    for subset in subsets:
        subset_name = subset.replace('.', '-')
        subset_dir = librispeech_dir / subset_name
        
        print(f"\nValidating subset: {subset_name}")
        
        # Check if subset directory exists
        if not subset_dir.exists():
            print(f"  ✗ Subset directory not found: {subset_dir}")
            all_valid = False
            continue
        
        # Count audio and transcript files
        audio_files = list(subset_dir.rglob("*.flac"))
        transcript_files = list(subset_dir.rglob("*.trans.txt"))
        
        audio_count = len(audio_files)
        transcript_count = len(transcript_files)
        
        print(f"  ✓ Directory exists: {subset_dir}")
        print(f"  ✓ Audio files (.flac): {audio_count}")
        print(f"  ✓ Transcript files (.trans.txt): {transcript_count}")
        
        if audio_count == 0:
            print(f"  ✗ No audio files found!")
            all_valid = False
        
        if transcript_count == 0:
            print(f"  ✗ No transcript files found!")
            all_valid = False
        
        # Check sample audio file integrity
        if audio_files:
            sample_audio = audio_files[0]
            try:
                data, samplerate = sf.read(str(sample_audio))
                duration = len(data) / samplerate
                print(f"  ✓ Sample audio valid: {sample_audio.name} ({duration:.2f}s, {samplerate}Hz)")
            except Exception as e:
                print(f"  ✗ Sample audio corrupted: {sample_audio.name} - {e}")
                all_valid = False
        
        # Check sample transcript file
        if transcript_files:
            sample_transcript = transcript_files[0]
            try:
                with open(sample_transcript, 'r') as f:
                    lines = f.readlines()
                    print(f"  ✓ Sample transcript valid: {sample_transcript.name} ({len(lines)} utterances)")
            except Exception as e:
                print(f"  ✗ Sample transcript corrupted: {sample_transcript.name} - {e}")
                all_valid = False
        
        total_audio_files += audio_count
        total_transcript_files += transcript_count
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ Validation PASSED")
    else:
        print("✗ Validation FAILED")
    print(f"{'='*60}")
    print(f"Total audio files: {total_audio_files}")
    print(f"Total transcript files: {total_transcript_files}")
    
    return all_valid


def test_sample_audio(librispeech_dir: Path, subsets: list, save_sample: bool = True):
    """
    Download and test a sample audio file from the dataset.
    Optionally saves the sample and creates a waveform visualization.
    
    Args:
        librispeech_dir: Path to LibriSpeech directory
        subsets: List of subsets
        save_sample: If True, save sample audio and waveform to disk
    """
    print(f"\n{'='*60}")
    print("Testing Sample Audio")
    print(f"{'='*60}")
    
    for subset in subsets[:1]:  # Test first subset only
        subset_name = subset.replace('.', '-')
        subset_dir = librispeech_dir / subset_name
        
        # Find first audio file
        audio_files = list(subset_dir.rglob("*.flac"))
        if not audio_files:
            print(f"No audio files found in {subset_name}")
            continue
        
        sample_audio = audio_files[0]
        
        # Read audio
        try:
            data, samplerate = sf.read(str(sample_audio))
            duration = len(data) / samplerate
            
            print(f"\n✓ Sample Audio Details:")
            print(f"  File: {sample_audio.relative_to(librispeech_dir)}")
            print(f"  Sample rate: {samplerate} Hz")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Channels: {data.shape[-1] if data.ndim > 1 else 1}")
            print(f"  Data type: {data.dtype}")
            print(f"  Min amplitude: {np.min(data):.4f}")
            print(f"  Max amplitude: {np.max(data):.4f}")
            print(f"  Mean amplitude: {np.mean(data):.4f}")
            
            # Find corresponding transcript
            utterance_id = sample_audio.stem
            speaker_id, chapter_id, _ = utterance_id.split('-')
            transcript_file = sample_audio.parent / f"{speaker_id}-{chapter_id}.trans.txt"
            
            transcript_text = None
            if transcript_file.exists():
                with open(transcript_file, 'r') as f:
                    for line in f:
                        if line.startswith(utterance_id):
                            transcript_text = line.strip().split(' ', 1)[1]
                            print(f"\n✓ Transcript:")
                            print(f"  Utterance ID: {utterance_id}")
                            print(f"  Text: {transcript_text}")
                            break
            
            # Save sample for manual inspection
            if save_sample:
                samples_dir = librispeech_dir.parent / "samples"
                samples_dir.mkdir(exist_ok=True)
                
                # Copy sample audio
                sample_copy = samples_dir / f"sample_{subset_name}.flac"
                sf.write(str(sample_copy), data, samplerate)
                print(f"\n✓ Saved sample audio to: {sample_copy}")
                
                # Save transcript
                if transcript_text:
                    transcript_copy = samples_dir / f"sample_{subset_name}.txt"
                    with open(transcript_copy, 'w') as f:
                        f.write(f"Utterance ID: {utterance_id}\n")
                        f.write(f"Subset: {subset_name}\n")
                        f.write(f"Duration: {duration:.2f}s\n")
                        f.write(f"Sample rate: {samplerate}Hz\n\n")
                        f.write(f"Transcript:\n{transcript_text}\n")
                    print(f"✓ Saved transcript to: {transcript_copy}")
                
                # Generate waveform visualization
                if MATPLOTLIB_AVAILABLE:
                    try:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        
                        # Waveform
                        time = np.arange(len(data)) / samplerate
                        ax1.plot(time, data, linewidth=0.5)
                        ax1.set_xlabel('Time (seconds)')
                        ax1.set_ylabel('Amplitude')
                        ax1.set_title(f'Waveform: {utterance_id}')
                        ax1.grid(True, alpha=0.3)
                        
                        # Spectrogram
                        ax2.specgram(data, Fs=samplerate, cmap='viridis')
                        ax2.set_xlabel('Time (seconds)')
                        ax2.set_ylabel('Frequency (Hz)')
                        ax2.set_title('Spectrogram')
                        
                        # Add transcript as text
                        if transcript_text:
                            fig.text(0.5, 0.02, f'Transcript: "{transcript_text}"', 
                                   ha='center', fontsize=10, wrap=True)
                        
                        plt.tight_layout()
                        
                        plot_path = samples_dir / f"sample_{subset_name}_waveform.png"
                        plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print(f"✓ Saved waveform visualization to: {plot_path}")
                        print(f"\n  You can view the waveform and spectrogram by opening:")
                        print(f"    {plot_path}")
                        
                    except Exception as e:
                        print(f"⚠ Could not generate waveform plot: {e}")
                else:
                    print(f"\n  Note: Install matplotlib to generate waveform visualizations:")
                    print(f"    pip install matplotlib")
                
                print(f"\n  To listen to the sample (on a machine with audio):")
                print(f"    # Using ffplay (from ffmpeg):")
                print(f"    ffplay -autoexit {sample_copy}")
                print(f"    # Using aplay:")
                print(f"    aplay {sample_copy}")
                print(f"    # Using Python:")
                print(f"    python -c 'import soundfile as sf; import sounddevice as sd; data, sr = sf.read(\"{sample_copy}\"); sd.play(data, sr); sd.wait()'")
            
            print(f"\n✓ Sample audio test PASSED")
            
        except Exception as e:
            print(f"\n✗ Sample audio test FAILED: {e}")
            return False
    
    return True


def download_librispeech_hf(data_dir: str, subsets: list, validate: bool = True, save_samples: bool = True):
    """
    Download LibriSpeech dataset from HuggingFace and save in openslr.org format.
    
    Args:
        data_dir: Directory to save the dataset
        subsets: List of subsets to download (e.g., ['train.clean.100', 'dev.clean'])
        validate: Whether to validate the dataset after download
        save_samples: Whether to save sample audio files for inspection
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
    
    # Validate dataset if requested
    if validate:
        if not validate_dataset(librispeech_dir, subsets):
            print("\n⚠ Validation failed - dataset may be incomplete")
            sys.exit(1)
        
        # Test a sample audio file
        test_sample_audio(librispeech_dir, subsets, save_sample=save_samples)
    
    print(f"\n{'='*60}")
    print("✓ All checks passed!")
    print(f"{'='*60}")
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
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation after download'
    )
    
    parser.add_argument(
        '--no-save-samples',
        action='store_true',
        help='Do not save sample audio files and visualizations'
    )
    
    args = parser.parse_args()
    
    # Convert subset names if needed (train-clean-100 -> train.clean.100)
    subsets = [s.replace('-', '.') if '-' in s else s for s in args.subsets]
    
    download_librispeech_hf(
        args.data_dir, 
        subsets, 
        validate=not args.no_validate,
        save_samples=not args.no_save_samples
    )


if __name__ == "__main__":
    main()

