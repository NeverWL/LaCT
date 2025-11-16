# -*- coding: utf-8 -*-

from .configuration_lact_asr import LaCTASRConfig
from .modeling_lact_asr import LaCTASRModel, LaCTASRForCTC
from .audio_features import AudioFeatureExtractor, MelSpectrogramExtractor
from .layer_lact_asr import LaCTASRLayer

# Hybrid model imports (optional - may require transformers)
try:
    from .configuration_hybrid_asr import HybridLaCTWav2Vec2Config
    from .modeling_hybrid_asr import HybridLaCTWav2Vec2Model, HybridLaCTWav2Vec2ForCTC
    from .wav2vec2_encoder import Wav2Vec2FeatureEncoder, ModifiedWav2Vec2Encoder
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False
    # Define dummy classes if transformers not available
    HybridLaCTWav2Vec2Config = None
    HybridLaCTWav2Vec2Model = None
    HybridLaCTWav2Vec2ForCTC = None
    Wav2Vec2FeatureEncoder = None
    ModifiedWav2Vec2Encoder = None

__all__ = [
    "LaCTASRConfig",
    "LaCTASRModel", 
    "LaCTASRForCTC",
    "AudioFeatureExtractor",
    "MelSpectrogramExtractor",
    "LaCTASRLayer",
]

if HYBRID_AVAILABLE:
    __all__.extend([
        "HybridLaCTWav2Vec2Config",
        "HybridLaCTWav2Vec2Model",
        "HybridLaCTWav2Vec2ForCTC",
        "Wav2Vec2FeatureEncoder",
        "ModifiedWav2Vec2Encoder",
    ])
