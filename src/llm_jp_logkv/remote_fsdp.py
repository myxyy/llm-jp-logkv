"""Student-only FSDP2 and a bounded, synchronous dedicated-teacher protocol.

The final global rank is the teacher; ranks [0, world_size-1) are students.
Load project settings before importing this module in standalone programs.
"""
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, register_fsdp_forward_method
from torch.distributed.tensor import DTensor

from .distillation import DistillationModel


class RemoteStudent(DistillationModel):
    """Own only student modules; source is a temporary CPU initialization model."""

    def __init__(self, source, **kwargs):
        super().__init__(source, attention_training='full', **kwargs)
        # Do not mutate source layers: tests/callers may still own the source.
        layers = []
        for original in self.student.layers:
            layer = nn.Module()
            for name in ('input_layernorm', 'post_attention_layernorm', 'mlp'):
                setattr(layer, name, getattr(original, name))
            layers.append(layer)
        self.student.layers = nn.ModuleList(layers)
        object.__delattr__(self, 'teacher')

    def forward(self, token_ids, lengths, teacher_hidden):
        return self.loss_from_teacher_hidden(token_ids, lengths, teacher_hidden)


def shard_student(model, group):
    mesh = DeviceMesh.from_group(group, 'cuda')
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for attention in model.student.attentions:
        fully_shard(attention, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        register_fsdp_forward_method(attention, 'forward_full')
    for conv in model.student.causal_convs:
        fully_shard(conv, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    # The root controls backward state; frozen FFNs/head/embeddings stay replicated.
    frozen = {p for p in model.parameters() if not p.requires_grad}
    fully_shard(model, mesh=mesh, ignored_params=frozen, mp_policy=policy,
                reshard_after_forward=True)
    assert all(isinstance(p, DTensor) for p in model.parameters() if p.requires_grad)
    return model


def clip_sharded_grad_norm(parameters, group, max_norm=1.):
    """Norm over all shards, not one rank's local subset or replicated norms."""
    parameters = list(parameters)
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    for p in parameters:
        if p.grad is None:
            raise RuntimeError('Missing trainable gradient')
        total += p.grad.to_local().double().square().sum()
    dist.all_reduce(total, group=group)
    norm = total.sqrt()
    if not torch.isfinite(norm).item():
        raise FloatingPointError('Non-finite sharded gradient norm')
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.)
    for p in parameters:
        p.grad.to_local().mul_(scale)
    return norm


class TeacherChannel:
    """One request per synchronous student microbatch; no teacher autograd graph.

    All requests have the same padded shape. The teacher processes one student's
    batch at a time, so five students do not imply a five-times-larger teacher
    activation peak. The world barrier separates world P2P and student collectives.
    """
    FORWARD, STOP = 1, 2

    def __init__(self, device, students, hidden_size):
        self.device, self.students, self.hidden_size = device, students, hidden_size
        self.teacher_rank = students

    def command(self, code, batch=0, length=0):
        header = torch.tensor([code, batch, length], device=self.device, dtype=torch.int64)
        dist.broadcast(header, src=0)
        return header.tolist()

    @torch.no_grad()
    def hidden(self, inputs):
        inputs = inputs.contiguous()
        batch, length = inputs.shape
        self.command(self.FORWARD, batch, length)
        dist.send(inputs, dst=self.teacher_rank)
        result = torch.empty(batch, length, self.hidden_size, dtype=torch.bfloat16, device=self.device)
        dist.recv(result, src=self.teacher_rank)
        dist.barrier()
        return result

    @torch.no_grad()
    def serve(self, teacher):
        while True:
            code, batch, length = self.command(0)
            if code == self.STOP:
                return
            if code != self.FORWARD or batch < 1 or length < 1:
                raise RuntimeError('Invalid teacher command')
            # Receive all inputs before any student enters another process group.
            batches = []
            for rank in range(self.students):
                inputs = torch.empty(batch, length, dtype=torch.long, device=self.device)
                dist.recv(inputs, src=rank)
                batches.append(inputs)
            for rank, inputs in enumerate(batches):
                hidden = teacher.model(inputs, use_cache=False).last_hidden_state.contiguous()
                dist.send(hidden, dst=rank)
                del hidden
            dist.barrier()


def full_adapter_state(student, rank):
    """Gather one parameter at a time; only rank zero retains full CPU copies."""
    result = {}
    for name, p in student.named_parameters():
        if p.requires_grad:
            full = p.full_tensor()
            if rank == 0:
                result[name] = full.detach().cpu()
            del full
    return result


def local_optimizer_state(optimizer):
    """Plain CPU tensors only, safe for weights_only load and a new process group."""
    def convert(value):
        if isinstance(value, DTensor):
            return {'_fsdp_local': value.to_local().detach().cpu(),
                    'shape': tuple(value.shape), 'stride': tuple(value.stride())}
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(optimizer.state_dict())


def restore_local_optimizer(optimizer, state):
    """Restore exactly the same shard topology and parameter ordering."""
    saved_ids = [i for group in state['param_groups'] for i in group['params']]
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    if len(saved_ids) != len(parameters):
        raise ValueError('Optimizer parameter count differs')
    for index, p in zip(saved_ids, parameters):
        for key, value in state['state'].get(index, {}).items():
            if isinstance(value, dict) and '_fsdp_local' in value:
                if tuple(value['shape']) != tuple(p.shape) or value['_fsdp_local'].shape != p.to_local().shape:
                    raise ValueError('Optimizer shard shape differs')
                state['state'][index][key] = DTensor.from_local(
                    value['_fsdp_local'].to(p.device), p.device_mesh, p.placements,
                    shape=torch.Size(value['shape']), stride=tuple(value['stride']))
    optimizer.load_state_dict(state)
