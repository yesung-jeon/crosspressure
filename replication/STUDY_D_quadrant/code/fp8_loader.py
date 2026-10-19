# Copied 2026-10-14 from ../../../../silicon_neutrality/code/fp8_loader.py for Study D (unchanged below this line).
# Copied 2026-10-07 from ../agent/code/fp8_loader.py (same author session). W8A8 FP8; fidelity vs BF16 checked there (TV 0.013, 24 prompts).
"""Load a HF causal LM with W8A8 FP8 linear layers (per-output-channel weight scale,
per-token dynamic activation scale) on GPUs where torch._scaled_mm only supports
tensorwise scaling (e.g. RTX 50xx on Windows). Row/column scales applied after the matmul,
which is mathematically identical to rowwise scaling."""
import json, os, glob, torch
from torch import nn

FP8_MAX = 448.0


class FP8Linear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        w = weight.to('cuda', torch.float32)
        s = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        self.register_buffer('wq_t', (w / s).to(torch.float8_e4m3fn).t())  # column-major view for _scaled_mm
        self.register_buffer('w_scale_bf16', s.t().to(torch.bfloat16))     # (1, out)
        self.bias = None if bias is None else nn.Parameter(bias.to('cuda', torch.bfloat16), requires_grad=False)
        self.in_features, self.out_features = w.shape[1], w.shape[0]
        self._one = torch.ones((), device='cuda', dtype=torch.float32)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        n = x2.shape[0]
        pad = (-n) % 16  # _scaled_mm needs M divisible by 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        xs = x2.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / FP8_MAX
        xq = (x2.float() / xs).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.wq_t, scale_a=self._one, scale_b=self._one, out_dtype=torch.bfloat16)
        y.mul_(xs.to(torch.bfloat16)).mul_(self.w_scale_bf16)
        if pad:
            y = y[:n]
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)


def load_fp8(model_id, revision='main'):
    from transformers import AutoConfig, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path = snapshot_download(model_id, revision=revision, allow_patterns=['*.json', '*.safetensors'])
    config = AutoConfig.from_pretrained(path)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation='sdpa')
    linear_names = {n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n != 'lm_head'}
    pending = {}
    for shard in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        with safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                mod_name, _, leaf = key.rpartition('.')
                if mod_name in linear_names:
                    pending.setdefault(mod_name, {})[leaf] = t
                    if 'weight' in pending[mod_name] and ('bias' in pending[mod_name] or
                            model.get_submodule(mod_name).bias is None):
                        d = pending.pop(mod_name)
                        parent, _, child = mod_name.rpartition('.')
                        setattr(model.get_submodule(parent), child, FP8Linear(d['weight'], d.get('bias')))
                else:
                    parent_mod = model.get_submodule(mod_name)
                    old = getattr(parent_mod, leaf)
                    new = nn.Parameter(t.to('cuda', torch.bfloat16), requires_grad=False) if isinstance(old, nn.Parameter) \
                        else t.to('cuda', torch.bfloat16)
                    setattr(parent_mod, leaf, new)
                del t
    if pending:
        raise RuntimeError(f'Incomplete linear weights: {list(pending)[:3]}')
    if getattr(config, 'tie_word_embeddings', False):
        model.lm_head.weight = model.model.embed_tokens.weight
    # non-persistent buffers (rotary inv_freq) were created on meta: rebuild them on GPU
    for name, mod in list(model.named_modules()):
        if hasattr(mod, 'inv_freq') and mod.inv_freq.is_meta:
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent), child, type(mod)(config=mod.config, device=torch.device('cuda')))
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        if p.is_meta:
            raise RuntimeError(f'Meta tensor left: {n}')
    model.eval()
    from transformers import GenerationConfig
    try:
        model.generation_config = GenerationConfig.from_pretrained(path)
    except Exception:
        pass
    return model
