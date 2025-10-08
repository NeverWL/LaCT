# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple
import math


class MelSpectrogramExtractor(nn.Module):
    """
    Mel-spectrogram feature extractor for audio input.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,  # 10ms
        win_length: int = 400,  # 25ms
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = True,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        if f_max is None:
            f_max = float(sample_rate // 2)
            
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel-spectrogram features from audio waveform.
        
        Args:
            waveform: Audio waveform tensor [batch_size, num_samples]
            
        Returns:
            mel_spec: Mel-spectrogram [batch_size, n_mels, time_frames]
        """
        # Extract mel-spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        
        # Convert to log scale (add small epsilon to avoid log(0))
        mel_spec = torch.log(mel_spec + 1e-8)
        
        return mel_spec


class AudioFeatureExtractor(nn.Module):
    """
    Complete audio feature extraction pipeline with convolutional layers.
    Converts raw audio or mel-spectrograms to transformer-ready features.
    """
    
    def __init__(
        self,
        config,
        input_type: str = "mel",  # "mel" or "raw"
    ):
        super().__init__()
        
        self.config = config
        self.input_type = input_type
        
        if input_type == "mel":
            self.mel_extractor = MelSpectrogramExtractor(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
                n_mels=config.n_mels,
            )
            input_dim = config.n_mels
        else:
            # For raw audio, we'll use a 1D conv to extract features
            self.mel_extractor = None
            input_dim = 1
            
        # Convolutional layers to process audio features
        conv_layers = []
        current_dim = input_dim
        
        for i in range(config.audio_encoder_layers):
            out_channels = config.hidden_size if i == config.audio_encoder_layers - 1 else current_dim * 2
            conv_layers.extend([
                nn.Conv1d(
                    current_dim, 
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                # Use float32 for BatchNorm to prevent NaN in mixed precision training
                nn.BatchNorm1d(out_channels, dtype=torch.float32),
                nn.ReLU(),
                nn.Dropout(config.audio_encoder_dropout)
            ])
            
            if i < config.audio_encoder_layers - 1:
                current_dim = current_dim * 2
            else:
                current_dim = config.hidden_size
                
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Positional encoding for audio sequences
        self.positional_encoding = AudioPositionalEncoding(
            config.hidden_size, 
            max_len=config.max_position_embeddings
        )
        
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio input.
        
        Args:
            audio_input: Raw audio [batch_size, num_samples] or 
                        mel-spectrogram [batch_size, n_mels, time_frames]
                        
        Returns:
            features: Audio features [batch_size, seq_len, hidden_size]
        """
        if self.input_type == "mel" and self.mel_extractor is not None:
            if len(audio_input.shape) == 2:  # Raw audio input
                # Extract mel-spectrogram
                mel_features = self.mel_extractor(audio_input)  # [B, n_mels, T]
            else:  # Already mel-spectrogram
                mel_features = audio_input
        else:
            # Raw audio processing
            if len(audio_input.shape) == 2:
                mel_features = audio_input.unsqueeze(1)  # [B, 1, T]
            else:
                mel_features = audio_input
                
        # Apply convolutional layers
        conv_features = self.conv_layers(mel_features)  # [B, hidden_size, T]
        
        # Transpose to [B, T, hidden_size] for transformer
        features = conv_features.transpose(1, 2)
        
        # Add positional encoding
        features = self.positional_encoding(features)
        
        return features


class AudioPositionalEncoding(nn.Module):
    """
    Positional encoding for audio sequences.
    Uses sinusoidal encoding similar to transformers but adapted for audio.
    """
    
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input features.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            
        Returns:
            x: Features with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SpecAugment(nn.Module):
    """
    SpecAugment data augmentation for mel-spectrograms.
    Implements frequency masking, time masking, and time warping.
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
        prob: float = 0.5,
    ):
        super().__init__()
        
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.prob = prob
        
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram [batch_size, n_mels, time_frames]
            
        Returns:
            augmented_spec: Augmented mel-spectrogram
        """
        if not self.training or torch.rand(1) > self.prob:
            return mel_spec
            
        batch_size, n_mels, time_frames = mel_spec.shape
        augmented = mel_spec.clone()
        
        for b in range(batch_size):
            # Frequency masking
            for _ in range(self.num_freq_masks):
                freq_mask_size = torch.randint(0, min(self.freq_mask_param, n_mels), (1,))
                freq_mask_start = torch.randint(0, n_mels - freq_mask_size + 1, (1,))
                augmented[b, freq_mask_start:freq_mask_start + freq_mask_size, :] = 0
                
            # Time masking  
            for _ in range(self.num_time_masks):
                time_mask_size = torch.randint(0, min(self.time_mask_param, time_frames), (1,))
                time_mask_start = torch.randint(0, time_frames - time_mask_size + 1, (1,))
                augmented[b, :, time_mask_start:time_mask_start + time_mask_size] = 0
                
        return augmented
