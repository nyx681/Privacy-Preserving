"""Sum condition gradients, update once on rank 0, then broadcast parameters."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist

class World:
    def __init__(self,backend='nccl'):
        self.rank=int(os.environ.get('RANK',0));self.size=int(os.environ.get('WORLD_SIZE',1))
        self.local_rank=int(os.environ.get('LOCAL_RANK',0))
        self.device=torch.device('cuda',self.local_rank) if backend=='nccl' else torch.device('cpu')
        if backend=='nccl':torch.cuda.set_device(self.device)
        if self.size>1:dist.init_process_group(backend=backend,timeout=timedelta(minutes=60))
    def owns(self,index):return index%self.size==self.rank
    def barrier(self):
        if self.size>1:dist.barrier()
    def broadcast_object(self,obj):
        values=[obj if self.rank==0 else None]
        if self.size>1:dist.broadcast_object_list(values,src=0)
        return values[0]
    def gather(self,items):
        if self.size==1:return items
        result=[None]*self.size if self.rank==0 else None
        dist.gather_object(items,result,dst=0)
        return [x for part in result for x in part] if self.rank==0 else None
    def sum_gradients(self,grads):
        if self.size>1:
            for grad in grads:dist.reduce(grad,dst=0,op=dist.ReduceOp.SUM)
        return grads
    def sync_parameters(self,parameters):
        if self.size>1:
            with torch.no_grad():
                for p in parameters:dist.broadcast(p,src=0)
    def close(self):
        if self.size>1:dist.destroy_process_group()

def update_once(world,parameters,local_gradients,optimizer,epsilon):
    """Weights are applied before this call. Never divide by world size."""
    gradients=world.sum_gradients(local_gradients)
    if world.rank==0:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        for p,g in zip(parameters,gradients):
            assert torch.isfinite(g).all()
            p.grad=g
        optimizer.step()
        with torch.no_grad():
            for p in parameters:p.clamp_(-epsilon,epsilon)
    world.sync_parameters(parameters)
    return gradients
