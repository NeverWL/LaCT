# -*- coding: utf-8 -*-

from typing import Optional
import sys
import os

# Add parent directory to path to import from lact_llm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lact_llm'))

from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig


class LaCTASRConfig(LaCTSWIGLUConfig):
    """
    Configuration for LaCT-ASR model.
    Extends LaCTSWIGLUConfig with ASR-specific parameters.

    Args:
        audio_feature_dim (int): Dimension of input audio features (e.g., 80 for mel-spectrograms)
        sample_rate (int): Audio sample rate in Hz
        hop_length (int): Hop length for STFT
        win_length (int): Window length for STFT  
        n_mels (int): Number of mel filterbank channels
        n_fft (int): FFT size for STFT
        ctc_vocab_size (int): Vocabulary size for CTC (including blank token)
        use_ctc (bool): Whether to use CTC loss
        use_attention (bool): Whether to use attention-based decoder
        audio_encoder_layers (int): Number of convolutional layers before LaCT
        audio_encoder_dropout (float): Dropout rate in audio encoder
        ctc_blank_id (int): ID of the blank token for CTC
    """
    model_type = 'lact_asr'
    
    def __init__(
        self,
        # Audio-specific parameters
        audio_feature_dim: int = 80,  # Mel-spectrogram features
        sample_rate: int = 16000,
        hop_length: int = 160,  # 10ms at 16kHz
        win_length: int = 400,  # 25ms at 16kHz
        n_mels: int = 80,
        n_fft: int = 512,
        
        # ASR model parameters
        ctc_vocab_size: int = 32,  # Characters + blank
        use_ctc: bool = True,
        use_attention: bool = False,
        audio_encoder_layers: int = 2,
        audio_encoder_dropout: float = 0.1,
        ctc_blank_id: int = 0,
        
        # Audio-specific chunk and window sizes
        lact_chunk_size: int = 4096,  # Larger chunks for audio (40s at 100fps)
        window_size: int = 4096,
        max_position_embeddings: int = 16384,  # Support longer audio sequences
        
        # Override some LLM defaults for ASR
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attn_heads: int = 12,
        num_lact_heads: int = 4,
        vocab_size: int = 32,  # Will be overridden by ctc_vocab_size
        
        **kwargs,
    ):
        # Initialize parent class with vocab_size set to ctc_vocab_size
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attn_heads=num_attn_heads,
            num_lact_heads=num_lact_heads,
            lact_chunk_size=lact_chunk_size,
            window_size=window_size,
            max_position_embeddings=max_position_embeddings,
            vocab_size=ctc_vocab_size,
            **kwargs
        )
        
        # Audio-specific parameters
        self.audio_feature_dim = audio_feature_dim
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        # ASR model parameters
        self.ctc_vocab_size = ctc_vocab_size
        self.use_ctc = use_ctc
        self.use_attention = use_attention
        self.audio_encoder_layers = audio_encoder_layers
        self.audio_encoder_dropout = audio_encoder_dropout
        self.ctc_blank_id = ctc_blank_id
