# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import warnings

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers library not available. Wav2Vec2 encoder cannot be used.")


class Wav2Vec2FeatureEncoder(nn.Module):
    """
    Wav2Vec2 feature encoder adapted for hybrid LaCT + Wav2Vec2 model.
    
    The default wav2vec2 encoder has 20x downsampling (5 strides of 2x each).
    This encoder is modified to have 5x downsampling for higher temporal resolution.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        target_ds_factor: int = 5,  # Target downsampling factor (5x vs default 20x)
        freeze_pretrained: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for Wav2Vec2 encoder. "
                "Install it with: pip install transformers"
            )
        
        self.model_name = model_name
        self.target_ds_factor = target_ds_factor
        self.freeze_pretrained = freeze_pretrained
        self.dropout_rate = dropout
        
        # Load pretrained wav2vec2 model to extract feature encoder
        try:
            pretrained_model = Wav2Vec2Model.from_pretrained(model_name)
            pretrained_config = Wav2Vec2Config.from_pretrained(model_name)
            pretrained_encoder = pretrained_model.feature_extractor
        except Exception as e:
            warnings.warn(
                f"Could not load pretrained model {model_name}: {e}. "
                "Creating encoder from scratch."
            )
            pretrained_config = Wav2Vec2Config.from_pretrained(
                model_name, 
                trust_remote_code=True
            ) if hasattr(Wav2Vec2Config, 'from_pretrained') else None
            pretrained_encoder = None
        
        # Extract or create feature encoder layers
        if pretrained_encoder is not None:
            # Use pretrained feature encoder
            self.conv_layers = nn.ModuleList(list(pretrained_encoder.conv_layers))
            self.layer_norm = pretrained_encoder.layer_norm if hasattr(pretrained_encoder, 'layer_norm') else None
        else:
            # Create feature encoder from scratch matching wav2vec2 architecture
            # 7 convolutional layers as in wav2vec2 BASE
            conv_layers = []
            in_channels = 1
            
            # Wav2Vec2 BASE feature encoder structure
            # Layer 0: Conv1d(1, 512, kernel=10, stride=5, padding=0) -> 2x downsampling
            # Layers 1-6: Conv1d(512, 512, kernel=3, stride=2, padding=1) -> 2x each = 32x total
            # Default total: 2 * 2^6 = 128x downsampling
            
            # For 5x downsampling, we adjust the first layer stride
            # Option 1: First layer stride=5 (simple 5x)
            # Option 2: First layer stride=2, second stride=2, third stride=1, rest stride=1 (4x)
            # Option 3: First layer stride=5, rest stride=1 (5x)
            
            # We'll use Option 3: first layer with stride=5, rest with stride=1
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, 512, kernel_size=10, stride=target_ds_factor, padding=0),
                    nn.Dropout(dropout),
                )
            )
            
            for _ in range(6):  # 6 more layers
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),  # stride=1 instead of 2
                        nn.Dropout(dropout),
                    )
                )
            
            self.conv_layers = nn.ModuleList(conv_layers)
            self.layer_norm = nn.GroupNorm(num_groups=512, num_channels=512)
        
        # Calculate actual downsampling factor
        # First layer: stride=target_ds_factor, rest: stride=1
        self.actual_ds_factor = target_ds_factor
        
        # Freeze pretrained weights if requested
        if freeze_pretrained:
            for param in self.parameters():
                param.requires_grad = False
        
        # Dropout for encoder output
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from raw audio waveform.
        
        Args:
            waveform: Raw audio tensor [batch_size, num_samples]
            
        Returns:
            features: Encoded features [batch_size, seq_len, hidden_size]
        """
        # Add channel dimension if needed: [B, T] -> [B, 1, T]
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Apply convolutional layers
        hidden_states = waveform
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
            # Apply GELU activation (standard in wav2vec2)
            hidden_states = F.gelu(hidden_states)
        
        # Apply layer norm if available
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Transpose to [B, T, D] for transformer: [B, D, T] -> [B, T, D]
        features = hidden_states.transpose(1, 2)
        
        return features
    
    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output sequence length given input length.
        
        Args:
            input_length: Input audio length in samples
            
        Returns:
            output_length: Output feature sequence length
        """
        # First layer: kernel=10, stride=target_ds_factor, padding=0
        # Output length = (input_length - kernel_size) / stride + 1
        output_length = (input_length - 10) // self.target_ds_factor + 1
        
        # Subsequent layers: kernel=3, stride=1, padding=1 -> length preserved
        return output_length


class ModifiedWav2Vec2Encoder(nn.Module):
    """
    Alternative implementation that modifies existing wav2vec2 encoder layers
    to achieve 5x downsampling instead of 20x.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        target_ds_factor: int = 5,
        freeze_pretrained: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.target_ds_factor = target_ds_factor
        self.freeze_pretrained = freeze_pretrained
        
        # Load pretrained model
        try:
            pretrained_model = Wav2Vec2Model.from_pretrained(model_name)
            pretrained_encoder = pretrained_model.feature_extractor
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model {model_name}: {e}")
        
        # Copy and modify conv layers
        self.conv_layers = nn.ModuleList()
        
        for i, layer in enumerate(pretrained_encoder.conv_layers):
            conv = layer.conv if hasattr(layer, 'conv') else layer[0]
            
            if i == 0:
                # First layer: modify stride to achieve target downsampling
                # Original: stride=5 (in wav2vec2 BASE)
                # Modified: stride=target_ds_factor
                modified_conv = nn.Conv1d(
                    conv.in_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size[0],
                    stride=target_ds_factor,  # Modified stride
                    padding=0,
                    bias=conv.bias is not None,
                )
                # Copy pretrained weights
                with torch.no_grad():
                    modified_conv.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        modified_conv.bias.copy_(conv.bias)
                
                # Create new sequential layer
                new_layer = nn.Sequential(modified_conv)
                if hasattr(layer, 'activation'):
                    new_layer.add_module('activation', layer.activation)
                if hasattr(layer, 'layer_norm'):
                    new_layer.add_module('layer_norm', layer.layer_norm)
                if dropout > 0:
                    new_layer.add_module('dropout', nn.Dropout(dropout))
                
                self.conv_layers.append(new_layer)
            else:
                # Subsequent layers: modify stride from 2 to 1 to preserve temporal resolution
                # Original: stride=2 (in wav2vec2 BASE)
                # Modified: stride=1
                modified_conv = nn.Conv1d(
                    conv.in_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size[0],
                    stride=1,  # Modified stride (was 2)
                    padding=1,  # Adjust padding for stride=1
                    bias=conv.bias is not None,
                )
                # Copy pretrained weights
                with torch.no_grad():
                    modified_conv.weight.copy_(conv.weight)
                    if conv.bias is not None:
                        modified_conv.bias.copy_(conv.bias)
                
                # Create new sequential layer
                new_layer = nn.Sequential(modified_conv)
                if hasattr(layer, 'activation'):
                    new_layer.add_module('activation', layer.activation)
                if hasattr(layer, 'layer_norm'):
                    new_layer.add_module('layer_norm', layer.layer_norm)
                if dropout > 0:
                    new_layer.add_module('dropout', nn.Dropout(dropout))
                
                self.conv_layers.append(new_layer)
        
        # Copy layer norm if available
        self.layer_norm = None
        if hasattr(pretrained_encoder, 'layer_norm'):
            self.layer_norm = pretrained_encoder.layer_norm
        
        # Freeze if requested
        if freeze_pretrained:
            for param in self.parameters():
                param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Calculate actual downsampling factor
        self.actual_ds_factor = target_ds_factor  # Only first layer downsamples
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features from raw audio."""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        hidden_states = waveform
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
            # Apply GELU if not already applied
            if not any(isinstance(m, (nn.GELU, torch.nn.functional.GELU)) for m in layer):
                hidden_states = F.gelu(hidden_states)
        
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        
        hidden_states = self.dropout(hidden_states)
        
        # Transpose: [B, D, T] -> [B, T, D]
        features = hidden_states.transpose(1, 2)
        
        return features
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length."""
        # First layer: (input_length - 10) / stride + 1
        output_length = (input_length - 10) // self.target_ds_factor + 1
        # Subsequent layers preserve length (stride=1, padding=1)
        return output_length

