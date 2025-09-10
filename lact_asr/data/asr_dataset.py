# -*- coding: utf-8 -*-

import os
import json
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import re
from collections import Counter
import string

logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """
    Base ASR dataset class that can be subclassed for specific datasets.
    """
    
    def __init__(
        self,
        manifest_path: str,
        vocab_path: Optional[str] = None,
        sample_rate: int = 16000,
        max_duration: float = 20.0,  # Maximum audio duration in seconds
        min_duration: float = 0.5,   # Minimum audio duration in seconds
        normalize_text: bool = True,
        audio_transforms = None,
    ):
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.normalize_text = normalize_text
        self.audio_transforms = audio_transforms
        
        # Load manifest
        self.data = self._load_manifest(manifest_path)
        
        # Load or create vocabulary
        if vocab_path and os.path.exists(vocab_path):
            self.vocab = self._load_vocab(vocab_path)
        else:
            logger.info("Creating vocabulary from transcripts...")
            self.vocab = self._create_vocab_from_data()
            if vocab_path:
                self._save_vocab(vocab_path)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        
        # Filter data by duration
        self.data = self._filter_by_duration()
        
        logger.info(f"Loaded {len(self.data)} samples")
        logger.info(f"Vocabulary size: {len(self.vocab)}")
    
    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest file. Each line should be a JSON object with 'audio_filepath' and 'text'."""
        data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def _load_vocab(self, vocab_path: str) -> List[str]:
        """Load vocabulary from file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f if line.strip()]
        return vocab
    
    def _save_vocab(self, vocab_path: str):
        """Save vocabulary to file."""
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for char in self.vocab:
                f.write(f"{char}\n")
    
    def _create_vocab_from_data(self) -> List[str]:
        """Create character-level vocabulary from the dataset."""
        char_counter = Counter()
        
        for item in self.data:
            text = item['text']
            if self.normalize_text:
                text = self._normalize_text(text)
            char_counter.update(text)
        
        # Create vocabulary: blank token + most common characters
        vocab = ['<blank>']  # CTC blank token
        
        # Add common characters in order of frequency
        for char, count in char_counter.most_common():
            if char not in vocab:
                vocab.append(char)
        
        return vocab
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for training."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _filter_by_duration(self) -> List[Dict]:
        """Filter samples by audio duration."""
        filtered_data = []
        
        for item in self.data:
            try:
                # Get audio info
                info = torchaudio.info(item['audio_filepath'])
                duration = info.num_frames / info.sample_rate
                
                if self.min_duration <= duration <= self.max_duration:
                    item['duration'] = duration
                    filtered_data.append(item)
                    
            except Exception as e:
                logger.warning(f"Error processing {item['audio_filepath']}: {e}")
                continue
        
        logger.info(f"Filtered {len(self.data) - len(filtered_data)} samples by duration")
        return filtered_data
    
    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of character indices."""
        if self.normalize_text:
            text = self._normalize_text(text)
        
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Handle unknown characters - could add <unk> token
                logger.warning(f"Unknown character: {char}")
        
        return indices
    
    def indices_to_text(self, indices: List[int]) -> str:
        """Convert list of indices back to text."""
        chars = []
        for idx in indices:
            if idx in self.idx_to_char and idx != 0:  # Skip blank token
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load audio
        waveform, orig_sr = torchaudio.load(item['audio_filepath'])
        
        # Resample if necessary
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Apply audio transforms if provided
        if self.audio_transforms:
            waveform = self.audio_transforms(waveform)
        
        # Convert text to indices
        text_indices = self.text_to_indices(item['text'])
        
        return {
            'audio': waveform.squeeze(0),  # Remove channel dimension
            'audio_length': waveform.shape[1],
            'text': item['text'],
            'text_indices': torch.tensor(text_indices, dtype=torch.long),
            'text_length': len(text_indices),
            'duration': item.get('duration', 0.0),
            'audio_filepath': item['audio_filepath']
        }


class LibriSpeechDataset(ASRDataset):
    """
    LibriSpeech dataset loader.
    Assumes LibriSpeech format with .flac files and corresponding .txt transcripts.
    """
    
    def __init__(
        self,
        root_dir: str,
        subset: str = "train-clean-100",  # train-clean-100, train-clean-360, etc.
        **kwargs
    ):
        self.root_dir = Path(root_dir)
        self.subset = subset
        
        # Create manifest if it doesn't exist
        manifest_path = self.root_dir / f"{subset}_manifest.json"
        if not manifest_path.exists():
            self._create_librispeech_manifest(manifest_path)
        
        super().__init__(str(manifest_path), **kwargs)
    
    def _create_librispeech_manifest(self, manifest_path: Path):
        """Create manifest file for LibriSpeech dataset."""
        logger.info(f"Creating LibriSpeech manifest for {self.subset}")
        
        subset_dir = self.root_dir / self.subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"LibriSpeech subset directory not found: {subset_dir}")
        
        manifest_data = []
        
        # Walk through LibriSpeech directory structure
        for speaker_dir in subset_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                
                # Find transcript file
                transcript_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not transcript_file.exists():
                    continue
                
                # Read transcripts
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) < 2:
                            continue
                        
                        file_id = parts[0]
                        transcript = parts[1]
                        
                        # Find corresponding audio file
                        audio_file = chapter_dir / f"{file_id}.flac"
                        if audio_file.exists():
                            manifest_data.append({
                                'audio_filepath': str(audio_file),
                                'text': transcript
                            })
        
        # Save manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in manifest_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created manifest with {len(manifest_data)} samples")


class CommonVoiceDataset(ASRDataset):
    """
    Mozilla Common Voice dataset loader.
    Assumes Common Voice format with .mp3/.wav files and .tsv metadata.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",  # train, dev, test
        language: str = "en",
        **kwargs
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.language = language
        
        # Create manifest if it doesn't exist
        manifest_path = self.root_dir / f"{language}_{split}_manifest.json"
        if not manifest_path.exists():
            self._create_commonvoice_manifest(manifest_path)
        
        super().__init__(str(manifest_path), **kwargs)
    
    def _create_commonvoice_manifest(self, manifest_path: Path):
        """Create manifest file for Common Voice dataset."""
        logger.info(f"Creating Common Voice manifest for {self.language} {self.split}")
        
        # Find TSV file
        tsv_file = self.root_dir / f"{self.split}.tsv"
        if not tsv_file.exists():
            raise FileNotFoundError(f"Common Voice TSV file not found: {tsv_file}")
        
        # Read TSV file
        df = pd.read_csv(tsv_file, sep='\t')
        
        manifest_data = []
        audio_dir = self.root_dir / "clips"
        
        for _, row in df.iterrows():
            audio_file = audio_dir / row['path']
            if audio_file.exists() and pd.notna(row['sentence']):
                manifest_data.append({
                    'audio_filepath': str(audio_file),
                    'text': row['sentence']
                })
        
        # Save manifest
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for item in manifest_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created manifest with {len(manifest_data)} samples")


class ASRDataCollator:
    """
    Data collator for ASR datasets that handles padding and batching.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        max_audio_length: Optional[int] = None,
        max_text_length: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract components
        audios = [item['audio'] for item in batch]
        audio_lengths = [item['audio_length'] for item in batch]
        text_indices = [item['text_indices'] for item in batch]
        text_lengths = [item['text_length'] for item in batch]
        
        # Determine max lengths for padding
        max_audio_len = max(audio_lengths)
        max_text_len = max(text_lengths)
        
        # Apply limits if specified
        if self.max_audio_length:
            max_audio_len = min(max_audio_len, self.max_audio_length)
        if self.max_text_length:
            max_text_len = min(max_text_len, self.max_text_length)
        
        # Pad audio
        padded_audios = []
        actual_audio_lengths = []
        
        for audio, length in zip(audios, audio_lengths):
            if length > max_audio_len:
                # Truncate if too long
                audio = audio[:max_audio_len]
                length = max_audio_len
            else:
                # Pad if too short
                padding = torch.zeros(max_audio_len - length)
                audio = torch.cat([audio, padding])
            
            padded_audios.append(audio)
            actual_audio_lengths.append(length)
        
        # Pad text
        padded_texts = []
        actual_text_lengths = []
        
        for text, length in zip(text_indices, text_lengths):
            if length > max_text_len:
                # Truncate if too long
                text = text[:max_text_len]
                length = max_text_len
            else:
                # Pad if too short
                padding = torch.full((max_text_len - length,), self.pad_token_id, dtype=torch.long)
                text = torch.cat([text, padding])
            
            padded_texts.append(text)
            actual_text_lengths.append(length)
        
        return {
            'audio_input': torch.stack(padded_audios),
            'input_lengths': torch.tensor(actual_audio_lengths, dtype=torch.long),
            'labels': torch.stack(padded_texts),
            'label_lengths': torch.tensor(actual_text_lengths, dtype=torch.long),
            'texts': [item['text'] for item in batch],  # Keep original texts
            'audio_filepaths': [item['audio_filepath'] for item in batch]
        }


def create_vocab_from_transcripts(
    transcript_files: List[str], 
    output_path: str,
    normalize: bool = True
) -> List[str]:
    """
    Create vocabulary from transcript files.
    
    Args:
        transcript_files: List of paths to transcript files
        output_path: Path to save vocabulary
        normalize: Whether to normalize text
        
    Returns:
        List of vocabulary characters
    """
    char_counter = Counter()
    
    for file_path in transcript_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if normalize:
                    text = text.lower()
                    text = re.sub(r"[^\w\s']", "", text)
                    text = re.sub(r'\s+', ' ', text)
                char_counter.update(text)
    
    # Create vocabulary
    vocab = ['<blank>']  # CTC blank token
    for char, count in char_counter.most_common():
        if char not in vocab:
            vocab.append(char)
    
    # Save vocabulary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(f"{char}\n")
    
    logger.info(f"Created vocabulary with {len(vocab)} characters")
    return vocab


def create_asr_dataloader(
    dataset: ASRDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    collate_fn: Optional[ASRDataCollator] = None,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader for ASR dataset.
    """
    if collate_fn is None:
        collate_fn = ASRDataCollator()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs
    )
