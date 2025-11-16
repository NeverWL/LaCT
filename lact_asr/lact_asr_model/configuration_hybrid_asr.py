# -*- coding: utf-8 -*-

from typing import Optional
import sys
import os

# Add parent directory to path to import from lact_llm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lact_llm'))

from lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from .configuration_lact_asr import LaCTASRConfig


class HybridLaCTWav2Vec2Config(LaCTASRConfig):
    """
    Configuration for Hybrid LaCT + Wav2Vec2 ASR model.
    Extends LaCTASRConfig with wav2vec2-specific parameters.
    """

    def __init__(
        self,
        # Wav2Vec2 Feature Encoder
        use_wav2vec2_encoder: bool = True,
        wav2vec2_model_name: str = "facebook/wav2vec2-base-960h",
        encoder_target_ds_factor: int = 5,  # Target downsampling factor (5x vs default 20x)
        freeze_encoder: bool = False,  # Freeze pretrained encoder
        encoder_lr_scale: float = 0.1,  # Lower LR for encoder during fine-tuning
        
        # LaCT Transformer (Enhanced from BASE)
        num_hidden_layers: int = 16,  # 16 layers vs 12 in BASE
        hidden_size: int = 768,
        intermediate_size: int = 3072,  # FFN dimension (same as wav2vec2 BASE)
        num_attn_heads: int = 8,  # Can be 8 (wav2vec2 BASE) or 12 (LaCT style)
        num_lact_heads: int = 4,  # Fast-weight heads
        lact_fw_hidden_dim: int = 1536,  # Fast-weight adapter hidden dim
        lact_chunk_size: int = 4096,
        window_size: int = 4096,
        
        # Fast-weight configuration
        fast_weight_lr_init: float = 0.01,  # softplus(const_lr_bias) = 0.01
        use_muon: bool = True,  # Use MuOn optimizer for fast weights
        use_momentum: bool = True,
        w0_w2_low_rank: int = -1,  # Full rank unless specified
        
        # Dropout and regularization
        hidden_dropout: float = 0.1,  # Transformer + encoder output dropout
        attention_dropout: float = 0.1,
        layerdrop: float = 0.05,  # BASE-style LayerDrop
        
        # TTT configuration
        enable_ttt: bool = True,  # Enable test-time training
        ttt_loss_type: str = "masked_prediction",  # "masked_prediction" or "entropy"
        ttt_mask_prob: float = 0.15,  # Mask probability for masked prediction
        ttt_steps: int = 1,  # Number of TTT steps per utterance
        ttt_chunk_size: Optional[int] = None,  # If None, use full utterance (up to 15s)
        
        # CTC and decoding
        ctc_vocab_size: int = 32,
        ctc_blank_id: int = 0,
        
        **kwargs,
    ):
        # Set inter_multi based on lact_fw_hidden_dim and hidden_size
        # lact_fw_hidden_dim = hidden_size * inter_multi (approximately)
        inter_multi = lact_fw_hidden_dim / (hidden_size // num_lact_heads)
        
        # Initialize parent config with LaCT-specific parameters
        super().__init__(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attn_heads=num_attn_heads,
            num_lact_heads=num_lact_heads,
            intermediate_size=intermediate_size,
            lact_chunk_size=lact_chunk_size,
            window_size=window_size,
            max_position_embeddings=16384,  # Support longer audio sequences
            ctc_vocab_size=ctc_vocab_size,
            ctc_blank_id=ctc_blank_id,
            use_muon=use_muon,
            use_momentum=use_momentum,
            w0_w2_low_rank=w0_w2_low_rank,
            inter_multi=inter_multi,
            **kwargs
        )
        
        # Wav2Vec2 encoder parameters
        self.use_wav2vec2_encoder = use_wav2vec2_encoder
        self.wav2vec2_model_name = wav2vec2_model_name
        self.encoder_target_ds_factor = encoder_target_ds_factor
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_scale = encoder_lr_scale
        
        # LaCT specific parameters
        self.lact_fw_hidden_dim = lact_fw_hidden_dim
        
        # Dropout and regularization
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layerdrop = layerdrop
        
        # TTT parameters
        self.enable_ttt = enable_ttt
        self.ttt_loss_type = ttt_loss_type
        self.ttt_mask_prob = ttt_mask_prob
        self.ttt_steps = ttt_steps
        # Default ttt_chunk_size: ~15s audio at 16kHz with downsampling factor
        if ttt_chunk_size is None:
            self.ttt_chunk_size = int(15.0 * 16000 / encoder_target_ds_factor)  # ~15s audio
        else:
            self.ttt_chunk_size = ttt_chunk_size
        
        # Fast-weight learning rate initialization
        self.fast_weight_lr_init = fast_weight_lr_init

