"""Mage-VL StreamMind Gate 模型元件。"""

# 本档由固定 revision 的上游实作拆分调整；来源：microsoft/Mage-VL，revision d88b153285f1633a61b2f693c59c8576693af185（Apache-2.0）。
from dataclasses import dataclass

import torch
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

        self.ssms = nn.ModuleList(
            [create_block(config.d_model, d_intermediate=0, layer_idx=i) for i in range(config.n_ssm)]
        )
        self.norm_fn = nn.LayerNorm(config.d_model)

    def forward(self, embeds, inference_params=None):
        hidden_states = embeds
        residual = None
        for ssm in self.ssms:
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=inference_params
            )
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
            loss = nn.CrossEntropyLoss(
                weight=torch.tensor([0.15, 0.85], device=shift_logits.device)
            )(shift_logits, shift_labels)
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
        targets = self.cls_net.cls_model.model.embed_tokens(
            target_ids.reshape(batch * time)
        )
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
    pre_net = PreNet(d_code=8, d_model=12).eval()
    post_net = PostNet(d_model=12, n_class=4).eval()
    features = torch.randn(2, 3, 8)
    hidden_states = torch.randn(2, 3, 12)
    with torch.inference_mode():
        engine_pre = pre_net(features)
        baseline_pre = F.leaky_relu(F.linear(features, pre_net.fc3.weight, pre_net.fc3.bias))
        engine_post = post_net(hidden_states)
        baseline_post = F.linear(F.leaky_relu(hidden_states), post_net.fc3.weight, post_net.fc3.bias)
    print("==========engine===========")
    print(f"PreNet={engine_pre[0, 0, :4].tolist()}")
    print(f"PostNet={engine_post[0, 0, :4].tolist()}")
    print("==========baseline==========")
    print(f"PreNet linear + LeakyReLU={baseline_pre[0, 0, :4].tolist()}")
    print(f"PostNet LeakyReLU + linear={baseline_post[0, 0, :4].tolist()}")
    torch.testing.assert_close(engine_pre, baseline_pre, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(engine_post, baseline_post, atol=1e-6, rtol=1e-6)
    print(f"StreamMind projection layers match their PyTorch baselines on {test_device}.")

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
    baseline_classifier = Qwen3ForCausalLM(classifier_config).eval()
    engine_classifier = Qwen3ForCausalLMCls(classifier_config).eval()
    engine_classifier.load_state_dict(baseline_classifier.state_dict(), strict=True)
    classifier_inputs = torch.randn(2, 3, classifier_config.hidden_size)
    with torch.inference_mode():
        engine_logits = engine_classifier(
            inputs_embeds=classifier_inputs,
            attention_mask=torch.ones(2, 3, dtype=torch.long),
        )["logits"]
        baseline_logits = baseline_classifier(
            inputs_embeds=classifier_inputs,
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            use_cache=False,
            return_dict=True,
        ).logits
    print("==========engine===========")
    print(f"Qwen3ForCausalLMCls={engine_logits[0, -1].tolist()}")
    print("==========baseline==========")
    print(f"Transformers Qwen3ForCausalLM={baseline_logits[0, -1].tolist()}")
    torch.testing.assert_close(engine_logits, baseline_logits, atol=1e-5, rtol=1e-5)

    try:
        video_mamba = VideoMamba(SSMConfig(d_model=8, n_ssm=1)).eval()
    except ModuleNotFoundError as error:
        if error.name != "mamba_ssm":
            raise
        print("VideoMamba comparison skipped: optional mamba_ssm dependency is unavailable.")
    else:
        mamba_inputs = torch.randn(1, 3, 8)
        with torch.inference_mode():
            engine_mamba = video_mamba(mamba_inputs)
            baseline_hidden = mamba_inputs
            baseline_residual = None
            for block in video_mamba.ssms:
                baseline_hidden, baseline_residual = block(
                    baseline_hidden,
                    baseline_residual,
                    inference_params=None,
                )
            baseline_residual = (
                baseline_hidden + baseline_residual
                if baseline_residual is not None
                else baseline_hidden
            )
            baseline_mamba = video_mamba.norm_fn(
                baseline_residual.to(dtype=video_mamba.norm_fn.weight.dtype)
            )
        print("==========engine===========")
        print(f"VideoMamba={engine_mamba[0, 0, :4].tolist()}")
        print("==========baseline==========")
        print(f"Mamba blocks + residual norm={baseline_mamba[0, 0, :4].tolist()}")
        torch.testing.assert_close(engine_mamba, baseline_mamba, atol=1e-5, rtol=1e-5)
