"""Exercise both cache collectives with different classes and an empty rank."""
import torch
import torch.distributed as dist
from back_prop.model_v4.risk import ResidualRiskBank


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(91+rank)
    radio, semantic = torch.randn(16,18,requires_grad=True), torch.randn(16,6,requires_grad=True)
    labels = torch.full((16,),float(rank%2))
    keys = [(f'train{rank}',i) for i in range(16)]
    bank=ResidualRiskBank(sparsity=2,pool_size=6,beam=3,min_samples=8)
    bank.allowed_cases={f'train{r}' for r in range(world)}
    for step in range(2):
        rows = 0 if step and rank == 0 else 16
        bank.observe_radiomics(radio[:rows],labels[:rows],keys[:rows])
        bank.observe_semantics(semantic[:rows],labels[:rows],keys[:rows])
        assert len(bank.memory)==16*world and int(bank.fit_count)==step+1
        assert int(bank.baseline_count)==step+1
        for value in (bank.weights,bank.intercepts,bank.radiomics_weights,bank.radiomics_intercept):
            copies=[torch.empty_like(value) for _ in range(world)]
            dist.all_gather(copies,value)
            assert all(torch.equal(copies[0],v) for v in copies)
        base=bank.baseline_logits(radio)
        z,_,_=bank(semantic,base)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,labels[:,None].expand_as(z))
        loss.backward()
        assert radio.grad is None and torch.isfinite(semantic.grad).all()
    if rank==0: print('PASS V4 two-stage DDP refits, empty-rank collectives, frozen heads, live gradients',flush=True)
    dist.destroy_process_group()


if __name__=='__main__': main()
