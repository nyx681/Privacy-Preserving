"""CPU/Gloo engineering test of the real reduction/update utilities, not model evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from distributed import World, update_once

def objective(parameters,index,step,kind):
    # Distinct dependencies exercise protected-only, key-only and paired gradients.
    indices=([0,1] if index==0 else [1] if index==5 else [0]) if kind=='image' else [0]
    value=sum((p*(index+1)+.07*(step+1)).square().mean() for j,p in enumerate(parameters) if j in indices)
    return value

def main():
    torch.set_num_threads(1);world=World('gloo');reports=[]
    try:
        for kind,weights in [('image',[.5]+[.1]*5),('text',[.5]+[.125]*4)]:
            shapes=[7,5] if kind=='image' else [7]
            ps=[torch.zeros(n,requires_grad=True) for n in shapes]
            optimizer=torch.optim.Adam(ps,lr=.01) if world.rank==0 else None
            refs=[torch.zeros(n,requires_grad=True) for n in shapes]
            serial_optimizer=torch.optim.Adam(refs,lr=.01) if world.rank==0 else None
            for step in range(4):
                local=[torch.zeros_like(p) for p in ps]
                for index,w in enumerate(weights):
                    if world.owns(index):
                        gs=torch.autograd.grad(objective(ps,index,step,kind),ps,allow_unused=True)
                        for acc,g in zip(local,gs):
                            if g is not None:acc.add_(g,alpha=w)
                expected=[]
                if world.rank==0:
                    serial_optimizer.zero_grad(set_to_none=True)
                    total=sum(w*objective(refs,i,step,kind) for i,w in enumerate(weights))
                    total.backward();expected=[p.grad.clone() for p in refs]
                    serial_optimizer.step()
                    with torch.no_grad():
                        for p in refs:p.clamp_(-.02,.02)
                actual=update_once(world,ps,local,optimizer,.02)
                if world.rank==0:
                    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
                    for a,b in zip(ps,refs):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
                    # Check optimizer states, not just scale-insensitive Adam's first update.
                    for p,r in zip(ps,refs):
                        for key in ['step','exp_avg','exp_avg_sq']:
                            torch.testing.assert_close(optimizer.state[p][key],serial_optimizer.state[r][key],rtol=1e-6,atol=1e-7)
                shared=world.broadcast_object([p.detach().tolist() for p in ps] if world.rank==0 else None)
                for p,values in zip(ps,shared):assert torch.equal(p.detach(),torch.tensor(values))
            gathered=world.gather([i for i in range(25) if world.owns(i)])
            if world.rank==0:
                assert sorted(gathered)==list(range(25))
                reports.append({'experiment':kind,'world_size':world.size,'serial_gradient_and_Adam_agree':True,
                                'idle_ranks_supported':world.size>len(weights),'evaluation_coverage_once':True})
        if world.rank==0:
            import json
            print(json.dumps({'kind':'CPU distributed engineering test, not actual Qwen performance','checks':reports},indent=2))
        world.barrier()
    finally:world.close()

if __name__=='__main__':main()
