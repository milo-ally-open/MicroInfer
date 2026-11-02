"""Mage-VL StreamMind Gate 模型组件。"""

# 本文件由固定 revision 的上游实现拆分调整；来源：microsoft/Mage-VL，revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from dataclasses import dataclass

import torch
import torch.utils.benchmark
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3 import Qwen3ForCausalLM


class PreNet(nn.Module):
    def __init__(self, d_code, d_model):
        super().__init__()
        self.fc3 = nn.Linear(d_code, d_model)

    def forward(self, x):
        return F.leaky_relu(self.fc3(x))


class PostNet(nn.Module):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.fc3 = nn.Linear(d_model, n_class)

    def forward(self, x):
        return self.fc3(F.leaky_relu(x))


@dataclass
class SSMConfig:
    d_model: int = 2560
    n_ssm: int = 1


class VideoMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        from mamba_ssm.models.mixer_seq_simple import create_block

        self.ssms = nn.ModuleList([create_block(config.d_model, d_intermediate=0, layer_idx=i) for i in range(config.n_ssm)])
        self.norm_fn = nn.LayerNorm(config.d_model)

    def forward(self, embeds, inference_params=None):
        hidden_states = embeds
        residual = None
        for ssm in self.ssms:
            hidden_states, residual = ssm(hidden_states, residual, inference_params=inference_params)
        residual = hidden_states + residual if residual is not None else hidden_states
        return self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))


class Qwen3ForCausalLMCls(Qwen3ForCausalLM):
    def forward(self, inputs_embeds=None, labels=None, attention_mask=None, **kwargs):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss = nn.CrossEntropyLoss(weight=torch.tensor([0.15, 0.85], device=shift_logits.device))(shift_logits, shift_labels)
        return {"loss": loss, "logits": logits}


class ClsNet(nn.Module):
    def __init__(self, hidden_size=2560, num_layers=4):
        super().__init__()
        config = Qwen3Config(
            vocab_size=2,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=12288,
            head_dim=128,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            attention_bias=False,
        )
        self.cls_model = Qwen3ForCausalLMCls(config)

    def forward(self, x, labels=None, attention_mask=None):
        return self.cls_model(inputs_embeds=x, labels=labels, attention_mask=attention_mask)


class StreamMindGate(nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.pre_net = PreNet(hidden_size, hidden_size)
        self.mamba_model = VideoMamba(SSMConfig(d_model=hidden_size))
        self.post_net = PostNet(hidden_size, hidden_size)
        self.cls_net = ClsNet(hidden_size=hidden_size, num_layers=4)

    def perception_tokens(self, vision_tokens):
        """Convert [B,T,P,D] visual patches to one EPFE token per time step."""
        x = vision_tokens.mean(dim=2)
        batch, time, dim = x.shape
        x = self.pre_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)
        x = self.mamba_model(x)
        x = self.post_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)
        return x

    def forward(self, vision_tokens, response_positions=None):
        """Return [B,T,2] silent/speak logits for every EPFE time step."""
        tokens = self.perception_tokens(vision_tokens)
        batch, time, dim = tokens.shape
        target_ids = torch.zeros(batch, time, dtype=torch.long, device=tokens.device)
        if response_positions is not None:
            target_ids[:, torch.as_tensor(response_positions, device=tokens.device) - 1] = 1
        targets = self.cls_net.cls_model.model.embed_tokens(target_ids.reshape(batch * time))
        pair = torch.stack((tokens.reshape(batch * time, dim), targets), dim=1)
        rotary = self.cls_net.cls_model.model.rotary_emb
        saved_inv_freq = rotary.inv_freq
        try:
            # Match the training checkpoint, where the full model (including
            # non-persistent Qwen3 RoPE buffers) was cast to BF16.
            rotary.inv_freq = rotary.inv_freq.to(pair.dtype)
            output = self.cls_net(
                pair,
                attention_mask=torch.ones(pair.shape[:2], device=pair.device),
            )
        finally:
            rotary.inv_freq = saved_inv_freq
        # Autoregressive shift: position 0 predicts the target token at
        # position 1, matching StreamMind's logits[..., :-1, :] evaluation.
        return output["logits"][:, 0].reshape(batch, time, 2)


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(test_device)

    from transformers import Qwen3Config
    from transformers.models.qwen3 import Qwen3ForCausalLM

    torch.manual_seed(71)
    baseline_pre = nn.Sequential(nn.Linear(8, 12), nn.LeakyReLU()).to(test_device).eval()
    enhanced_pre = PreNet(d_code=8, d_model=12).to(test_device).eval()
    enhanced_pre.fc3.load_state_dict(baseline_pre[0].state_dict(), strict=True)
    baseline_post = nn.Sequential(nn.LeakyReLU(), nn.Linear(12, 4)).to(test_device).eval()
    enhanced_post = PostNet(d_model=12, n_class=4).to(test_device).eval()
    enhanced_post.fc3.load_state_dict(baseline_post[1].state_dict(), strict=True)
    features = torch.randn(2, 3, 8, device=test_device)
    hidden_states = torch.randn(2, 3, 12, device=test_device)

    classifier_config = Qwen3Config(
        vocab_size=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        eos_token_id=None,
    )
    classifier_config.rope_theta = float(classifier_config.rope_parameters["rope_theta"])
    classifier_config._attn_implementation = "eager"
    baseline_classifier = Qwen3ForCausalLM(classifier_config).to(test_device).eval()
    enhanced_classifier = Qwen3ForCausalLMCls(classifier_config).to(test_device).eval()
    enhanced_classifier.load_state_dict(baseline_classifier.state_dict(), strict=True)
    classifier_inputs = torch.randn(2, 3, classifier_config.hidden_size, device=test_device)
    classifier_mask = torch.ones(2, 3, dtype=torch.long, device=test_device)

    enhanced_cls_net = ClsNet(hidden_size=32, num_layers=1).to(test_device).eval()
    baseline_cls_net = Qwen3ForCausalLM(enhanced_cls_net.cls_model.config).to(test_device).eval()
    baseline_cls_net.load_state_dict(enhanced_cls_net.cls_model.state_dict(), strict=True)
    cls_net_inputs = torch.randn(1, 2, 32, device=test_device)
    cls_net_mask = torch.ones(1, 2, dtype=torch.long, device=test_device)

    components = [
        ("StreamMind PreNet", baseline_pre, enhanced_pre, (features,), (features,), {}, {}, None, None, 1e-6, 1e-6, False),
        (
            "StreamMind PostNet",
            baseline_post,
            enhanced_post,
            (hidden_states,),
            (hidden_states,),
            {},
            {},
            None,
            None,
            1e-6,
            1e-6,
            False,
        ),
        (
            "Transformers Qwen3 classifier",
            baseline_classifier,
            enhanced_classifier,
            (),
            (),
            {"inputs_embeds": classifier_inputs, "attention_mask": classifier_mask, "use_cache": False, "return_dict": True},
            {"inputs_embeds": classifier_inputs, "attention_mask": classifier_mask},
            "logits",
            "logits",
            1e-5,
            1e-5,
            False,
        ),
        (
            "Transformers Qwen3 ClsNet",
            baseline_cls_net,
            enhanced_cls_net,
            (),
            (cls_net_inputs,),
            {"inputs_embeds": cls_net_inputs, "attention_mask": cls_net_mask, "use_cache": False, "return_dict": True},
            {"attention_mask": cls_net_mask},
            "logits",
            "logits",
            1e-5,
            1e-5,
            False,
        ),
    ]

    # Mamba 是 MageVL 的可选外部依赖；存在时加载固定 revision 的官方实现进行对照。
    try:
        import importlib.util
        import sys
        from pathlib import Path

        official_snapshot = Path(__file__).resolve().parents[5] / "checkpoints" / "microsoft-Mage-VL-d88b153"
        official_name = "_mage_vl_official_streammind_gate"
        official_spec = importlib.util.spec_from_file_location(official_name, official_snapshot / "streammind_gate.py")
        if official_spec is None or official_spec.loader is None:
            raise RuntimeError(f"Unable to load the official Mage-VL StreamMind source: {official_snapshot}")
        official = importlib.util.module_from_spec(official_spec)
        sys.modules[official_name] = official
        official_spec.loader.exec_module(official)

        baseline_video_mamba = official.VideoMamba(official.SSMConfig(d_model=8, n_ssm=1)).to(test_device).eval()
        enhanced_video_mamba = VideoMamba(SSMConfig(d_model=8, n_ssm=1)).to(test_device).eval()
        enhanced_video_mamba.load_state_dict(baseline_video_mamba.state_dict(), strict=True)
        baseline_gate = official.StreamMindGate(hidden_size=32).to(test_device).eval()
        enhanced_gate = StreamMindGate(hidden_size=32).to(test_device).eval()
        enhanced_gate.load_state_dict(baseline_gate.state_dict(), strict=True)
    except ModuleNotFoundError as error:
        if error.name not in {"mamba_ssm", "selective_scan_cuda"}:
            raise
        print(f"VideoMamba and StreamMindGate skipped: optional dependency {error.name!r} is unavailable.")
    else:
        mamba_inputs = torch.randn(1, 3, 8, device=test_device)
        vision_tokens = torch.randn(1, 3, 2, 32, device=test_device)
        components.extend(
            [
                (
                    "MageVL official VideoMamba",
                    baseline_video_mamba,
                    enhanced_video_mamba,
                    (mamba_inputs,),
                    (mamba_inputs,),
                    {},
                    {},
                    None,
                    None,
                    1e-5,
                    1e-5,
                    False,
                ),
                (
                    "MageVL official StreamMindGate",
                    baseline_gate,
                    enhanced_gate,
                    (vision_tokens,),
                    (vision_tokens,),
                    {"response_positions": [2]},
                    {"response_positions": [2]},
                    None,
                    None,
                    1e-5,
                    1e-5,
                    False,
                ),
            ]
        )

    profile_activities = [torch.profiler.ProfilerActivity.CPU]
    if test_device.type == "cuda":
        profile_activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_sort = "self_cuda_time_total" if test_device.type == "cuda" else "self_cpu_time_total"

    for (
        component_name,
        baseline,
        enhanced,
        baseline_args,
        enhanced_args,
        baseline_kwargs,
        enhanced_kwargs,
        baseline_selector,
        enhanced_selector,
        atol,
        rtol,
        compare_hidden_states,
    ) in components:
        baseline_output = None
        baseline_raw_output = None
        baseline_median = None
        for stage, module, args, kwargs, selector in (
            ("baseline", baseline, baseline_args, baseline_kwargs, baseline_selector),
            ("enhanced", enhanced, enhanced_args, enhanced_kwargs, enhanced_selector),
        ):
            print(f"================= {stage} =================")
            print("   test:")
            print(f"       component={component_name}")
            with torch.inference_mode():
                raw_output = module(*args, **kwargs)
            if isinstance(selector, str):
                output = raw_output[selector] if isinstance(raw_output, dict) else getattr(raw_output, selector)
            elif selector is not None:
                output = raw_output[selector]
            else:
                output = raw_output
            if stage == "baseline":
                torch.testing.assert_close(output, output, atol=atol, rtol=rtol)
                baseline_output, baseline_raw_output = output, raw_output
                output_tensor = output if isinstance(output, torch.Tensor) else output[0]
                print(f"       PASS, shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")
            else:
                torch.testing.assert_close(output, baseline_output, atol=atol, rtol=rtol)
                if compare_hidden_states:
                    torch.testing.assert_close(
                        tuple(raw_output.hidden_states), tuple(baseline_raw_output.hidden_states), atol=atol, rtol=rtol
                    )
                if isinstance(output, torch.Tensor):
                    max_abs_error = (output - baseline_output).abs().max().item()
                    output_tensor = output
                else:
                    max_abs_error = max(
                        (
                            (actual - expected).abs().max().item()
                            for actual, expected in zip(output, baseline_output)
                            if isinstance(actual, torch.Tensor)
                        ),
                        default=0.0,
                    )
                    output_tensor = output[0]
                print(f"       parity=PASS, max_abs_error={max_abs_error:.3e}, shape={tuple(output_tensor.shape)}")

            print("   bench:")
            with torch.inference_mode():
                bench = torch.utils.benchmark.Timer(
                    stmt="module(*args, **kwargs)",
                    globals={"module": module, "args": args, "kwargs": kwargs},
                ).blocked_autorange(min_run_time=0.2)
            if stage == "baseline":
                baseline_median = bench.median
                print(f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}")
            else:
                print(
                    f"       median_ms={bench.median * 1000:.4f}, iqr_ms={bench.iqr * 1000:.4f}, speedup={baseline_median / bench.median:.3f}x"
                )

            print("   profile:")
            with torch.inference_mode():
                with torch.profiler.profile(activities=profile_activities, record_shapes=True, profile_memory=True) as profile:
                    module(*args, **kwargs)
            if test_device.type == "cuda":
                torch.cuda.synchronize(test_device)
            print(profile.key_averages().table(sort_by=profile_sort, row_limit=12))
