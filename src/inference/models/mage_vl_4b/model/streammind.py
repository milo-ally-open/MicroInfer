"""Mage-VL StreamMind Gate 权重加载与视频片段门控。"""

# 本模块拆分自 microsoft/Mage-VL revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。

from pathlib import Path
import torch


class MageVLStreamMindMixin:
    """为 MageVLModel 提供可选的 StreamMind 特征与事件门控路径。"""

    def _load_streammind_gate(self):
        if self.streammind_gate is not None:
            return self.streammind_gate
        from safetensors.torch import load_file
        from ..layers.streammind_gate import StreamMindGate

        if not self._streammind_model_path:
            raise RuntimeError(
                "StreamMind gate path is unavailable. Load the model with "
                "MageVLForConditionalGeneration.from_pretrained()."
            )
        gate = StreamMindGate(self.config.text_config.hidden_size)
        gate_path = Path(self._streammind_model_path) / "streammind_gate.safetensors"
        if not gate_path.is_file():
            from huggingface_hub import hf_hub_download

            gate_path = Path(
                hf_hub_download(
                    repo_id=self._streammind_model_path,
                    filename="streammind_gate.safetensors",
                )
            )
        state = load_file(str(gate_path))
        gate.load_state_dict(state, strict=True)
        gate.to(
            device=next(self.visual.parameters()).device,
            dtype=next(self.visual.parameters()).dtype,
        ).eval()
        self.streammind_gate = gate
        return gate

    def _streammind_vision_tokens(self, pixel_values, image_grid_thw, patch_positions=None):
        pixel_values = pixel_values.type(self.visual.embeddings.patch_embedding.weight.dtype)
        rope = self.visual.video_rope
        saved_rope = (rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w)
        try:
            # The StreamMind checkpoint was trained after the whole OV model,
            # including non-persistent RoPE buffers, was cast to BF16.
            rope.inv_freq_t = rope.inv_freq_t.to(pixel_values.dtype)
            rope.inv_freq_h = rope.inv_freq_h.to(pixel_values.dtype)
            rope.inv_freq_w = rope.inv_freq_w.to(pixel_values.dtype)
            vision_output = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                patch_positions=patch_positions,
            )
        finally:
            rope.inv_freq_t, rope.inv_freq_h, rope.inv_freq_w = saved_rope
        merged = vision_output.last_hidden_state.reshape(-1, vision_output.last_hidden_state.shape[-1])
        merge = self.visual.spatial_merge_size
        time = int(image_grid_thw[:, 0].sum().item())
        patches_per_time = int(
            (image_grid_thw[0, 1].item() // merge)
            * (image_grid_thw[0, 2].item() // merge)
        )
        vision_tokens = merged.reshape(1, time, patches_per_time, -1)
        return vision_tokens

    def streammind_gate_forward(self, pixel_values, image_grid_thw, patch_positions=None):
        """Run the gate on one segment without changing the base LLM visual path."""
        vision_tokens = self._streammind_vision_tokens(
            pixel_values, image_grid_thw, patch_positions=patch_positions
        )
        return self._load_streammind_gate()(vision_tokens)

    def streammind_gate_forward_segments(self, segments):
        """Run EPFE continuously over a list of codec segments from one stream."""
        tokens = [
            self._streammind_vision_tokens(
                segment["pixel_values"],
                segment["image_grid_thw"],
                patch_positions=segment.get("patch_positions"),
            )
            for segment in segments
        ]
        lengths = [token.shape[1] for token in tokens]
        boundaries = torch.tensor(lengths).cumsum(0).tolist()
        return self._load_streammind_gate()(
            torch.cat(tokens, dim=1), response_positions=boundaries
        )


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)
    import importlib.util
    import sys
    import types
    from types import SimpleNamespace

    from .configuration_mage_vl import MageVLVisionConfig
    from .vision_tower import MageVLVisionPretrainedModel

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

    torch.manual_seed(75)
    config = MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=8,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
    )
    reference_config = official.MageVLVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_channels=3,
        image_size=8,
        patch_size=2,
        out_hidden_size=32,
        spatial_merge_size=2,
    )
    config._attn_implementation = "eager"
    reference_config._attn_implementation = "eager"
    engine_visual = MageVLVisionPretrainedModel(config).eval()
    baseline_visual = official.MageVLVisionPretrainedModel(reference_config).eval()
    engine_visual.load_state_dict(baseline_visual.state_dict(), strict=True)
    engine_owner = SimpleNamespace(visual=engine_visual)
    baseline_owner = SimpleNamespace(visual=baseline_visual)
    pixel_values = torch.randn(4, 3, 2, 2)
    image_grid_thw = torch.tensor([[1, 2, 2]])
    patch_positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
    engine_rope = engine_visual.video_rope
    baseline_rope = baseline_visual.video_rope
    engine_saved = (engine_rope.inv_freq_t.clone(), engine_rope.inv_freq_h.clone(), engine_rope.inv_freq_w.clone())
    baseline_saved = (
        baseline_rope.inv_freq_t.clone(),
        baseline_rope.inv_freq_h.clone(),
        baseline_rope.inv_freq_w.clone(),
    )
    with torch.inference_mode():
        engine_tokens = MageVLStreamMindMixin._streammind_vision_tokens(
            engine_owner, pixel_values, image_grid_thw, patch_positions=patch_positions
        )
        baseline_tokens = official.MageVLModel._streammind_vision_tokens(
            baseline_owner, pixel_values, image_grid_thw, patch_positions=patch_positions
        )
    torch.testing.assert_close(engine_tokens, baseline_tokens, atol=1e-6, rtol=1e-6)
    for current, saved in zip(
        (engine_rope.inv_freq_t, engine_rope.inv_freq_h, engine_rope.inv_freq_w), engine_saved
    ):
        torch.testing.assert_close(current, saved, atol=0, rtol=0)
    for current, saved in zip(
        (baseline_rope.inv_freq_t, baseline_rope.inv_freq_h, baseline_rope.inv_freq_w), baseline_saved
    ):
        torch.testing.assert_close(current, saved, atol=0, rtol=0)
    print("==========engine===========")
    print(f"shape={tuple(engine_tokens.shape)}, sample={engine_tokens[0, 0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"official StreamMind vision tokens={baseline_tokens[0, 0, 0, :4].tolist()}")
    print(f"Mage-VL StreamMind visual-token preparation and RoPE restoration match the official source on {test_device}.")
