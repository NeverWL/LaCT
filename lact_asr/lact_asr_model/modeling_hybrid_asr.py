# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# Add parent directory to path to import from lact_llm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../lact_llm'))

from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import GatedMLP as TransformerMLP
from fla.modules import RMSNorm

from .configuration_hybrid_asr import HybridLaCTWav2Vec2Config
from .layer_lact_asr import LaCTASRLayer
from .wav2vec2_encoder import ModifiedWav2Vec2Encoder, Wav2Vec2FeatureEncoder
from .audio_features import SpecAugment

logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


class HybridLaCTWav2Vec2Block(nn.Module):
    """
    Hybrid ASR block combining wav2vec2 features with LaCT fast-weight updates.
    Based on LaCTASRBlock but adapted for hybrid architecture.
    """

    def __init__(self, config: HybridLaCTWav2Vec2Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = LaCTASRLayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            qkv_bias=config.qkv_bias,
            attn_qk_norm=config.attn_qk_norm,
            qkv_silu=config.qkv_silu,
            lr_dim=config.lr_dim,
            use_muon=config.use_muon,
            ttt_prenorm=config.ttt_prenorm,
            ttt_nope=config.ttt_nope,
            lr_parameterization=config.lr_parameterization,
            learnable_ttt_scale=config.learnable_ttt_scale,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            w0_w2_low_rank=config.w0_w2_low_rank,
            use_momentum=config.use_momentum,
            ttt_loss_type=config.ttt_loss_type,
            fw_init_gain=config.fw_init_gain,
            audio_adapt=True,
            temporal_reduction=1,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu
        )
        
        # LayerDrop for regularization
        self.layerdrop = config.layerdrop

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[Any]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        # LayerDrop: randomly skip layer during training
        if self.training and self.layerdrop > 0:
            if torch.rand(1).item() < self.layerdrop:
                return (hidden_states,) + (None,) * (2 if output_attentions else 1)

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        
        # Apply dropout to MLP output
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = F.dropout(hidden_states, p=self.config.hidden_dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class HybridLaCTWav2Vec2PreTrainedModel(PreTrainedModel):
    """
    Base class for Hybrid LaCT + Wav2Vec2 ASR models.
    """

    config_class = HybridLaCTWav2Vec2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['HybridLaCTWav2Vec2Block']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
        
        if isinstance(module, LaCTASRLayer):
            # Initialize the parameters of the model
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)

            logger.info(f"in PreTrainedModel initialize fast weights for LaCTASRLayer")
            # init w0, w1, w2
            if module.w0_w2_low_rank > 0:
                module.w0._init_weights()
                module.w2._init_weights()
            else:
                nn.init.normal_(module.w0, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))
                nn.init.normal_(module.w2, mean=0.0, std=1.0 / math.sqrt(module.fw_head_dim))
            
            nn.init.normal_(module.w1, mean=0.0, std=1.0/math.sqrt(module.d_h))


class HybridLaCTWav2Vec2Model(HybridLaCTWav2Vec2PreTrainedModel):
    """
    Hybrid LaCT + Wav2Vec2 model for ASR tasks.
    
    Architecture:
    1. Pretrained Wav2Vec2 7-layer CNN feature encoder (modified stride for 5x downsampling)
    2. 16 LaCT Transformer blocks with fast-weight adapters
    3. CTC decoder head
    """

    def __init__(
        self,
        config: HybridLaCTWav2Vec2Config
    ) -> 'HybridLaCTWav2Vec2Model':
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.config = config

        # Wav2Vec2 feature encoder (pretrained, modified stride)
        if config.use_wav2vec2_encoder:
            self.feature_encoder = ModifiedWav2Vec2Encoder(
                model_name=config.wav2vec2_model_name,
                target_ds_factor=config.encoder_target_ds_factor,
                freeze_pretrained=config.freeze_encoder,
                dropout=config.hidden_dropout,
            )
        else:
            # Fallback to standard audio feature extractor
            from .audio_features import AudioFeatureExtractor
            self.feature_encoder = AudioFeatureExtractor(
                config,
                input_type="raw"
            )
        
        # Optional SpecAugment for training
        if hasattr(config, 'use_spec_augment') and config.use_spec_augment:
            self.spec_augment = SpecAugment(
                freq_mask_param=15,
                time_mask_param=35,
                num_freq_masks=1,
                num_time_masks=1,
                prob=0.5,
            )
        else:
            self.spec_augment = None

        # LaCT transformer layers (16 layers)
        self.layers = nn.ModuleList([
            HybridLaCTWav2Vec2Block(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(
            config.hidden_size, 
            eps=config.norm_eps
        )
        
        # Dropout for encoder output
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        audio_input: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract audio features if not provided
        if audio_features is None:
            if audio_input is None:
                raise ValueError("Either audio_input or audio_features must be provided")
            
            # Extract features using wav2vec2 encoder
            hidden_states = self.feature_encoder(audio_input)  # [B, T, D]
        else:
            hidden_states = audio_features

        # Apply dropout to encoder output
        hidden_states = self.dropout(hidden_states)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        # Pass through LaCT transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    **kwargs
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class HybridLaCTWav2Vec2ForCTC(HybridLaCTWav2Vec2PreTrainedModel):
    """
    Hybrid LaCT + Wav2Vec2 model for ASR with CTC head for sequence-to-sequence learning.
    """

    def __init__(self, config: HybridLaCTWav2Vec2Config):
        super().__init__(config)
        self.model = HybridLaCTWav2Vec2Model(config)
        self.ctc_head = nn.Linear(config.hidden_size, config.ctc_vocab_size, bias=False)
        
        # CTC loss function
        self.ctc_loss = nn.CTCLoss(blank=config.ctc_blank_id, reduction='mean', zero_infinity=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model

    def forward(
        self,
        audio_input: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_lengths: Optional[torch.LongTensor] = None,
        input_lengths: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through the model
        outputs = self.model(
            audio_input=audio_input,
            audio_features=audio_features,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        
        # Apply CTC head to get logits
        logits = self.ctc_head(hidden_states)
        
        # Apply log softmax for CTC
        log_probs = torch.log_softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            if input_lengths is None:
                # Calculate input lengths from hidden states
                if hidden_states.size(1) > 0:
                    input_lengths = torch.full(
                        (hidden_states.size(0),), 
                        hidden_states.size(1), 
                        dtype=torch.long, 
                        device=hidden_states.device
                    )
                else:
                    raise ValueError("Cannot determine input_lengths from empty hidden_states")
            
            if label_lengths is None:
                label_lengths = torch.full(
                    (labels.size(0),), 
                    labels.size(1), 
                    dtype=torch.long, 
                    device=labels.device
                )
            
            # Transpose for CTC loss: [batch, time, vocab] -> [time, batch, vocab]
            log_probs_transposed = log_probs.transpose(0, 1)
            
            loss = self.ctc_loss(
                log_probs_transposed,
                labels,
                input_lengths,
                label_lengths
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

