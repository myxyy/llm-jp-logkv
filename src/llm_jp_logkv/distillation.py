"""Trainable adapters and exact, chunked full-vocabulary distillation loss.

The frozen teacher and student share only frozen base modules. Trainable LoRA
or full projection weights belong to the student, never to the teacher.
"""
import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .llama_model import LlamaLogKVForCausalLM


@torch.no_grad()
def teacher_hidden_states(teacher, inputs, *, include_blocks=False):
    """Final-normalized states and optional raw decoder-block outputs.

    Hooks capture the last block BEFORE the final model RMSNorm, unlike using
    the last entry of Transformers' hidden_states. Copy into one bounded buffer
    rather than retaining a list of teacher activations or stacking a second copy.
    """
    if not include_blocks:
        return teacher.model(inputs, use_cache=False).last_hidden_state
    blocks = None
    handles = []

    def capture(index, module, args, output):
        nonlocal blocks
        hidden = output[0] if isinstance(output, tuple) else output
        if blocks is None:
            blocks = hidden.new_empty((len(teacher.model.layers), *hidden.shape))
        blocks[index].copy_(hidden)

    try:
        for index, layer in enumerate(teacher.model.layers):
            handles.append(layer.register_forward_hook(partial(capture, index)))
        final = teacher.model(inputs, use_cache=False).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    return {'final': final, 'blocks': blocks}


def loss_metrics(totals, ce_weight, hidden_mse_weight=0.):
    """Decode globally summed statistics; token/layer means match the loss."""
    count = totals[2]
    result = {'kl': (totals[0]/count).item(), 'ce': (totals[1]/count).item()}
    loss = totals[0] + ce_weight*totals[1]
    if totals.numel() > 3:
        layers = (totals.numel()-3)//2
        mse = totals[3:3+layers]/count
        energy = totals[3+layers:]/count
        result.update(hidden_mse=mse.mean().item(),
                      hidden_mse_by_layer=mse.tolist(), teacher_rms_by_layer=energy.sqrt().tolist(),
                      weighted_hidden_mse=(hidden_mse_weight*mse.mean()).item())
        loss = loss + hidden_mse_weight*totals[3:3+layers].mean()
    result['loss'] = (loss/count).item()
    return result


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


class FullAttentionLinear(nn.Module):
    """Independent pretrained Q/K/V/O weights with FP32 master parameters.

    Cast for the current computation as in LoRA, without retaining a frozen
    base projection inside this module or modifying the teacher's parameters.
    """

    def __init__(self, base):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = nn.Parameter(base.weight.detach().to(dtype=torch.float32, copy=True))
        self.bias = (None if base.bias is None else
                     nn.Parameter(base.bias.detach().to(dtype=torch.float32, copy=True)))

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


def add_full_attention(student):
    for attention in student.attentions:
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(attention, name, FullAttentionLinear(getattr(attention, name)))


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
                 checkpoint_layers=True, no_rope=False, conv_kernel_size=0,
                 attention_training='lora', hidden_mse_weight=0.):
        super().__init__()
        if attention_training not in ('lora', 'full'):
            raise ValueError('attention_training must be lora or full')
        if temperature <= 0 or ce_weight < 0 or loss_chunk_tokens <= 0:
            raise ValueError('Invalid loss settings')
        if not math.isfinite(hidden_mse_weight) or hidden_mse_weight < 0:
            raise ValueError('hidden_mse_weight must be finite and nonnegative')
        teacher.requires_grad_(False)
        teacher.eval()
        # Do not register a second alias of the whole teacher in this module.
        object.__setattr__(self, 'teacher', teacher)
        self.student = LlamaLogKVForCausalLM(teacher, mode='logkv', chunk_size=chunk_size,
                                           no_level_bias=False, no_rope=no_rope,
                                           conv_kernel_size=conv_kernel_size)
        self.student.attention_training = attention_training
        if attention_training == 'lora':
            add_lora(self.student, rank, alpha)
        else:
            add_full_attention(self.student)
        for attention in self.student.attentions:
            attention.core.recompute_attention = False  # outer layer checkpoint handles it
        self.loss_chunk_tokens = loss_chunk_tokens
        self.temperature, self.ce_weight = temperature, ce_weight
        self.checkpoint_layers = checkpoint_layers
        self.hidden_mse_weight = hidden_mse_weight

    @property
    def stats_size(self):
        return 3 + (2*len(self.student.layers) if self.hidden_mse_weight else 0)

    def _block(self, index, hidden, cos, sin):
        layer, attention = self.student.layers[index], self.student.attentions[index]
        if self.student.conv_kernel_size:
            hidden = self.student.causal_convs[index](hidden)
        hidden = hidden + attention.forward_full(layer.input_layernorm(hidden), (cos, sin))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _block_with_mse(self, index, hidden, cos, sin, teacher_hidden, mask):
        hidden = self._block(index, hidden, cos, sin)
        # Keep FP32 differences inside the layer checkpoint: backward recomputes
        # them instead of retaining every layer's large FP32 difference tensor.
        target = teacher_hidden.float()
        mse = ((hidden.float()-target).square().mean(-1)*mask).sum()
        energy = (target.square().mean(-1)*mask).sum()
        return hidden, mse, energy

    def student_hidden(self, input_ids, teacher_blocks=None, mask=None):
        hidden = self.student.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=hidden.device)[None]
        cos, sin = self.student.position_embeddings(hidden, pos)
        layer_stats = []
        for index in range(len(self.student.layers)):
            if teacher_blocks is not None:
                args = (index, hidden, cos, sin, teacher_blocks[index], mask)
                if self.checkpoint_layers and torch.is_grad_enabled():
                    hidden, mse, energy = checkpoint(self._block_with_mse, *args,
                        use_reentrant=False, preserve_rng_state=False)
                else:
                    hidden, mse, energy = self._block_with_mse(*args)
                layer_stats.append(torch.stack((mse, energy)))
                continue
            if self.checkpoint_layers and torch.is_grad_enabled():
                hidden = checkpoint(partial(self._block, index), hidden, cos, sin,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = self._block(index, hidden, cos, sin)
        hidden = self.student.norm(hidden)
        return (hidden, torch.stack(layer_stats)) if teacher_blocks is not None else hidden

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
        teacher_hidden = teacher_hidden_states(self.teacher, inputs,
                                               include_blocks=bool(self.hidden_mse_weight))
        return self.loss_from_teacher_hidden(token_ids, lengths, teacher_hidden)

    def loss_from_teacher_hidden(self, token_ids, lengths, teacher_hidden):
        """Exact vocabulary loss and optional MSE from detached teacher states."""
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        teacher_blocks = None
        if self.hidden_mse_weight:
            if not isinstance(teacher_hidden, dict) or set(teacher_hidden) != {'final', 'blocks'}:
                raise ValueError('Hidden MSE requires final and all block teacher states')
            teacher_blocks = teacher_hidden['blocks'].detach()
            if teacher_blocks.shape != (len(self.student.layers), *inputs.shape, self.student.config.hidden_size):
                raise ValueError('Teacher block hidden shape does not match the student')
            teacher_hidden = teacher_hidden['final']
        if teacher_hidden.shape != (*inputs.shape, self.student.config.hidden_size):
            raise ValueError('Teacher hidden shape does not match the input batch')
        teacher_hidden = teacher_hidden.detach()
        positions = torch.arange(inputs.shape[1], device=inputs.device)[None]
        mask = positions < (lengths[:, None] - 1)
        if self.hidden_mse_weight:
            hidden, layer_stats = self.student_hidden(inputs, teacher_blocks, mask)
        else:
            hidden = self.student_hidden(inputs)
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
        metrics = torch.stack((stats[0].detach(), stats[1].detach(), mask.sum().float()))
        if self.hidden_mse_weight:
            loss_sum = loss_sum + self.hidden_mse_weight*layer_stats[:, 0].mean()
            metrics = torch.cat((metrics, layer_stats.detach().T.flatten()))
        return loss_sum, metrics


def student_from_checkpoint(source, path, *, model_id, model_revision, chunk_size=4):
    """Restore LoRA or full attention and optional conv, leaving the base intact."""
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
                             conv_kernel_size=config.get('conv_kernel_size',0),
                             attention_training=config.get('attention_training','lora'))
    load_adapter(trained.student,torch.load(path/'adapter.pt',map_location='cpu',weights_only=True))
    return trained.student.eval()
