from llm_jp_logkv.settings import load_settings
load_settings()
import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(local)
dist.init_process_group('nccl',timeout=timedelta(seconds=60),device_id=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
x=torch.tensor(float(rank),device=f'cuda:{local}')
dist.all_reduce(x)
assert x.item()==world*(world-1)/2
if rank==0:
    print(f'NCCL all-reduce passed on {world} GPUs: sum={x.item()}',flush=True)
dist.barrier();dist.destroy_process_group()
