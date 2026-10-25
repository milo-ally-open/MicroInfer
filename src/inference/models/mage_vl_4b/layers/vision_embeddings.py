"""Mage-VL 视觉 Patch Embedding 与空间合并投影。"""

# 本模块沿用 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）实现。

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ..model.configuration_mage_vl import MageVLVisionConfig


class MageVLVisionEmbeddings(nn.Module):
    """
    Patch embedding layer that converts pre-processed patches to embeddings.

    This module is designed to receive patches that have already been extracted
    and arranged by the Qwen2VL image processor in 2x2 block spatial order.

    Input format: [total_patches, num_channels, patch_size, patch_size]
    Output format: [total_patches, embed_dim]
    """

    def __init__(self, config: MageVLVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        hidden_states = self.patch_embedding(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)

        return hidden_states


class MageVLVisionPatchMerger(nn.Module):
    """
    Patch merger that merges spatial_merge_size x spatial_merge_size patches into one.

    This module is designed to work with Qwen2VL-style patch processing where patches
    are already arranged in 2x2 block order by the image processor.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        layer_norm_eps: float = 1e-05,
        use_patch_position_encoding: bool = False,
        patch_position_encoding_type: str = "absolute",
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm(context_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )
        self.spatial_merge_size = spatial_merge_size
        self.use_patch_position_encoding = use_patch_position_encoding
        self.patch_position_encoding_type = patch_position_encoding_type

        if self.use_patch_position_encoding:
            if self.patch_position_encoding_type != "absolute":
                raise ValueError(
                    f"Unknown patch_position_encoding_type: {self.patch_position_encoding_type}. "
                    "Only 'absolute' is supported."
                )
            self.pos_emb_h = nn.Embedding(max_position_embeddings, dim)
            self.pos_emb_w = nn.Embedding(max_position_embeddings, dim)

    def forward(self, x: torch.Tensor, patch_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Merge patches from Qwen2VL-style input.

        The input patches are already arranged in 2x2 block order by the image processor,
        so we simply need to apply LayerNorm, reshape, and project through MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
               where seq_len = t * h * w (patches in 2x2 block order)

        Returns:
            Merged tensor of shape [batch_size, seq_len // spatial_merge_size^2, dim]
            or [seq_len // spatial_merge_size^2, dim]
        """
        if patch_positions is not None and patch_positions.dim() == 3:
            patch_positions = patch_positions.squeeze(0)

        x = self.ln_q(x).view(-1, self.hidden_size)
        x = self.mlp(x)

        if self.use_patch_position_encoding and patch_positions is not None:
            pp = patch_positions.view(-1, self.spatial_merge_size**2, 3)
            pp = pp[:, 0, :]
            pp = (pp // self.spatial_merge_size).long()

            x = x + self.pos_emb_h(pp[:, 1]) + self.pos_emb_w(pp[:, 2])

        return x


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from pathlib import Path

    official_snapshot = Path(__file__).resolve().parents[5] / "checkpoints" / "microsoft-Mage-VL-d88b153"
    official_package_name = "_mage_vl_official_d88b153"
    official_package = types.ModuleType(official_package_name)
    official_package.__path__ = [str(official_snapshot)]
    sys.modules[official_package_name] = official_package
    official_name = f"{official_package_name}.modeling_mage_vl"
    official_spec = importlib.util.spec_from_file_location(official_name, official_snapshot / "modeling_mage_vl.py")
    if official_spec is None or official_spec.loader is None:
        raise RuntimeError(f"Unable to load the official Mage-VL source at {official_snapshot}.")
    official = importlib.util.module_from_spec(official_spec)
    sys.modules[official_name] = official
    official_spec.loader.exec_module(official)

    torch.manual_seed(63)
    config = MageVLVisionConfig(
        hidden_size=64, intermediate_size=128, num_channels=3, image_size=8, patch_size=2
    )
    reference_config = official.MageVLVisionConfig(
        hidden_size=64, intermediate_size=128, num_channels=3, image_size=8, patch_size=2
    )
    engine_embeddings = MageVLVisionEmbeddings(config).eval()
    baseline_embeddings = official.MageVLVisionEmbeddings(reference_config).eval()
    engine_embeddings.load_state_dict(baseline_embeddings.state_dict(), strict=True)
    patches = torch.randn(4, 3, 2, 2)

    engine_merger = MageVLVisionPatchMerger(
        dim=32,
        context_dim=64,
        spatial_merge_size=2,
        use_patch_position_encoding=True,
        max_position_embeddings=8,
    ).eval()
    baseline_merger = official.MageVLVisionPatchMerger(
        dim=32,
        context_dim=64,
        spatial_merge_size=2,
        use_patch_position_encoding=True,
        max_position_embeddings=8,
    ).eval()
    engine_merger.load_state_dict(baseline_merger.state_dict(), strict=True)
    patch_positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    with torch.inference_mode():
        engine_embeds = engine_embeddings(patches)
        baseline_embeds = baseline_embeddings(patches)
        engine_merged = engine_merger(engine_embeds, patch_positions=patch_positions)
        baseline_merged = baseline_merger(baseline_embeds, patch_positions=patch_positions)
    torch.testing.assert_close(engine_embeds, baseline_embeds, atol=0, rtol=0)
    torch.testing.assert_close(engine_merged, baseline_merged, atol=0, rtol=0)
    print("==========engine===========")
    print(f"patches={tuple(engine_embeds.shape)}, merged={tuple(engine_merged.shape)}")
    print("==========baseline==========")
    print(f"official patches={tuple(baseline_embeds.shape)}, merged={tuple(baseline_merged.shape)}")
    print(f"Mage-VL patch embedding and merger match the official source on {test_device}.")
