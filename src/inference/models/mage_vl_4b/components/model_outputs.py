"""Mage-VL 輸出資料結構，對齊固定上游 revision。"""

# 本檔由固定 revision 的上游實作拆分調整；來源：microsoft/Mage-VL，revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers import AutoModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    replace_return_docstrings,
)
from transformers.utils.generic import is_flash_attention_requested

from .configuration_mage_vl import MageVLConfig, MageVLVisionConfig


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Mage-VL outputs, with hidden states and attentions.
    """
)
class MageVLModelOutputWithPast(ModelOutput):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Mage-VL causal language model (or autoregressive) outputs.
    """
)
class MageVLCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# ---------------------------------------------------------------------------
# Vision Rotary Embedding
# ---------------------------------------------------------------------------
