"""True GT-free inference, followed by one-to-one official-mask matching."""
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from back_prop.evaluate.eval_package.types import MaskCrop
from back_prop.evaluate.eval_package.comparisons import _crop_overlap
from back_prop.common.rng import preserve_rng


def missed_nodules(output, batch, *, mask_threshold=.5, minimum_iou=.1):
    predictions = [MaskCrop(((logit.float().sigmoid() >= mask_threshold) & valid).cpu().numpy(),
                            origin.cpu().numpy()) for logit,valid,origin in
                   zip(output.fine_mask_logits,output.roi_valid,output.crop_origins)]
    truth = [MaskCrop(mask.cpu().numpy().astype(bool),origin.cpu().numpy()) for mask,origin in
             zip(batch['target_mask_crops'],batch['target_mask_origins'])]
    scores = np.zeros((len(predictions), len(truth)))
    for i,pred in enumerate(predictions):
        for j,gt in enumerate(truth):
            scores[i,j] = _crop_overlap(pred,gt)['iou']
    matched = set()
    if scores.size:
        valid = (scores>0) & (scores>=minimum_iou)
        row,col = linear_sum_assignment(-(valid*(min(scores.shape)+1)+scores))
        matched = {int(j) for i,j in zip(row,col) if valid[i,j]}
    return {str(int(nodule)): j not in matched for j,nodule in enumerate(batch['nodule_ids'])}


@torch.inference_mode()
def mine_training(model, dataset, args, rank, world, completed_epoch):
    started = time.monotonic()
    was_training = model.training
    local = {}
    with preserve_rng():
        model.eval()
        for index in range(rank,len(dataset),world):
            batch = dataset[index]
            with torch.autocast('cuda',dtype=getattr(torch,args.amp_dtype)):
                output = model(batch['image'],image_hu=batch['image_hu'],batch=None,
                    object_threshold=args.mining_object_threshold,segmentation_only=True,update_bank=False)
            if output.teacher_forced_count or output.fallback_count:
                raise AssertionError('Mining must never route using official GT')
            local[batch['case_id']] = missed_nodules(output,batch,
                mask_threshold=args.mining_mask_threshold,minimum_iou=args.mining_minimum_iou)
            print(json.dumps(dict(event='mining_case',rank=rank,epoch=completed_epoch,
                case_id=batch['case_id'],misses=sum(local[batch['case_id']].values()),
                gt=len(batch['nodule_ids']),predictions=len(output.refined_query_indices))),flush=True)
            del output,batch
        gathered = [None]*world if world>1 else [local]
        if world>1:
            dist.all_gather_object(gathered,local)
        records = {}
        for group in gathered:
            if set(records)&set(group):
                raise AssertionError('Duplicate CT in distributed mining')
            records.update(group)
        model.train(was_training)
    summary = dict(epoch=completed_epoch,cases=len(records),
        nodules=sum(len(v) for v in records.values()),misses=sum(sum(v.values()) for v in records.values()),
        hard_cases=sum(any(v.values()) for v in records.values()),seconds=time.monotonic()-started,
        object_threshold=args.mining_object_threshold,mask_threshold=args.mining_mask_threshold,
        minimum_iou=args.mining_minimum_iou,gt_routing=False,training_only=True)
    if rank==0:
        (args.output_dir/f'mining_{completed_epoch:03d}.json').write_text(
            json.dumps(dict(summary=summary,records=records),indent=2)+'\n')
    return records,summary
