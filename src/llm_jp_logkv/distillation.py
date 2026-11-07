"""Trainable adapters and exact, chunked full-vocabulary distillation loss.

The frozen teacher and student share only frozen base modules. LoRA wrappers
are installed on the student's attention adapter objects, never on the teacher.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .llama_model import LlamaLogKVForCausalLM


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base = base
        self.base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features,
                                             device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank,
                                             device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        extra = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))
        return self.base(x) + extra * self.scaling


def add_lora(student, rank=32, alpha=64):
    for attention in student.attentions:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(attention, name, LoRALinear(getattr(attention, name), rank, alpha))


def adapter_state(student):
    return {name: p.detach().cpu().clone() for name, p in student.named_parameters()
            if p.requires_grad}


def load_adapter(student, state):
    expected = {name: p for name, p in student.named_parameters() if p.requires_grad}
    if set(expected) != set(state):
        raise ValueError('Adapter keys do not match the constructed model')
    with torch.no_grad():
        for name, p in expected.items():
            if p.shape != state[name].shape:
                raise ValueError(f'Adapter shape mismatch: {name}')
            p.copy_(state[name].to(p.device, p.dtype))


class DistillationModel(nn.Module):
    def __init__(self, teacher, *, rank=32, alpha=64, chunk_size=4,
                 loss_chunk_tokens=64, temperature=1., ce_weight=.1,
                 checkpoint_layers=True, no_rope=False, conv_kernel_size=0):
        super().__init__()
        if temperature <= 0 or ce_weight < 0 or loss_chunk_tokens <= 0:
            raise ValueError('Invalid loss settings')
        teacher.requires_grad_(False)
        teacher.eval()
        # Do not register a second alias of the whole teacher in this module.
        object.__setattr__(self, 'teacher', teacher)
        self.student = LlamaLogKVForCausalLM(teacher, mode='logkv', chunk_size=chunk_size,
                                           no_level_bias=False, no_rope=no_rope,
                                           conv_kernel_size=conv_kernel_size)
        add_lora(self.student, rank, alpha)
        for attention in self.student.attentions:
            attention.core.recompute_attention = False  # outer layer checkpoint handles it
        self.loss_chunk_tokens = loss_chunk_tokens
        self.temperature, self.ce_weight = temperature, ce_weight
        self.checkpoint_layers = checkpoint_layers

    def _block(self, index, hidden, cos, sin):
        layer, attention = self.student.layers[index], self.student.attentions[index]
        if self.student.conv_kernel_size:
            hidden = self.student.causal_convs[index](hidden)
        hidden = hidden + attention.forward_full(layer.input_layernorm(hidden), (cos, sin))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def student_hidden(self, input_ids):
        hidden = self.student.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=hidden.device)[None]
        cos, sin = self.student.position_embeddings(hidden, pos)
        for index in range(len(self.student.layers)):
            if self.checkpoint_layers and torch.is_grad_enabled():
                hidden = checkpoint(partial(self._block, index), hidden, cos, sin,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = self._block(index, hidden, cos, sin)
        return self.student.norm(hidden)

    def _loss_chunk(self, student_hidden, teacher_hidden, targets, mask):
        # Compute and normalize over the ENTIRE vocabulary, with only a bounded
        # number of sequence positions alive at once. This is not top-k KL.
        with torch.no_grad():
            teacher_logits = self.student.lm_head(teacher_hidden).float() / self.temperature
            teacher_logp = F.log_softmax(teacher_logits, dim=-1)
            teacher_p = teacher_logp.exp()
        logits = self.student.lm_head(student_hidden).float()
        student_logp = F.log_softmax(logits / self.temperature, dim=-1)
        kl = (teacher_p * (teacher_logp - student_logp)).sum(-1) * self.temperature ** 2
        if self.temperature == 1.:
            ce = F.nll_loss(student_logp.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            reduction='none').view_as(targets)
        else:
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                 reduction='none').view_as(targets)
        return torch.stack(((kl * mask).sum(), (ce * mask).sum()))

    def forward(self, token_ids, lengths):
        # Right padding only. Causality guarantees that unscored padding cannot
        # affect a preceding valid token, for either teacher or LogKV student.
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        with torch.no_grad():
            teacher_hidden = self.teacher.model(inputs, use_cache=False).last_hidden_state
        hidden = self.student_hidden(inputs)
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
        mask = positions < (lengths[:, None] - 1)
        stats = hidden.new_zeros(2, dtype=torch.float32)
        for start in range(0, inputs.shape[1], self.loss_chunk_tokens):
            end = min(inputs.shape[1], start + self.loss_chunk_tokens)
            args = (hidden[:, start:end], teacher_hidden[:, start:end],
                    targets[:, start:end], mask[:, start:end])
            if torch.is_grad_enabled():
                chunk = checkpoint(self._loss_chunk, *args, use_reentrant=False,
                                   preserve_rng_state=False)
            else:
                chunk = self._loss_chunk(*args)
            stats = stats + chunk
        loss_sum = stats[0] + self.ce_weight * stats[1]
        return loss_sum, torch.stack((stats[0].detach(), stats[1].detach(), mask.sum().float()))


def student_from_checkpoint(source, path, *, model_id, model_revision, chunk_size=4):
    """Restore student architecture and LoRA/conv weights, leaving the base intact."""
    import json
    from pathlib import Path
    from .inference import file_hash
    path=Path(path)
    config=json.loads((path/'config.json').read_text())
    complete=json.loads((path/'complete.json').read_text())
    if config['model_id']!=model_id or config['model_revision']!=model_revision:
        raise ValueError('Checkpoint base model/revision differs')
    if config['chunk_size']!=chunk_size or config['no_level_bias']:
        raise ValueError('Checkpoint attention configuration differs')
    if file_hash(path/'adapter.pt')!=complete['adapter_sha256']:
        raise ValueError('Adapter checksum mismatch')
    trained=DistillationModel(source,rank=config['rank'],alpha=config['alpha'],chunk_size=chunk_size,
                             no_rope=config.get('no_rope',False),
                             conv_kernel_size=config.get('conv_kernel_size',0))
    load_adapter(trained.student,torch.load(path/'adapter.pt',map_location='cpu',weights_only=True))
    return trained.student.eval()
