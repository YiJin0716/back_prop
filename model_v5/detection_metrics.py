"""Prediction-only candidate metrics, with GT used only for scoring."""
import torch
from scipy.optimize import linear_sum_assignment

from back_prop.common.matcher import match_coarse,pairwise_box_iou
from back_prop.common.roi import paste_compact_mask


@torch.no_grad()
def detection_counts(objects,boxes,masks,batch,selected,roi_shape=(128,128,128),minimum_coverage=.8):
    """Counts for aggregation across scans.

    Detection tolerance is half the largest GT box dimension, at least 3 mm.
    Matching maximizes the number of in-tolerance one-to-one detections before
    minimizing distance. Predictions near excluded annotations are ignored.
    """
    gt=batch['target_boxes'].to(boxes)
    shape=boxes.new_tensor(batch['isotropic_shape'])
    all_match=match_coarse(objects,boxes,masks,gt,batch['target_masks'])
    q,t=all_match.query_indices,all_match.target_indices
    probability=masks[q].float().sigmoid()
    target=batch['target_masks'].to(probability)[t]
    ignored=batch.get('ignored_target_masks')
    if ignored is not None and len(ignored):
        support=~ignored.to(probability.device).bool().any(0)[None] | target.bool()
        probability,target=probability*support,target*support
    p,g=probability.flatten(1),target.flatten(1)
    dice=2*(p*g).sum(1)/(p.sum(1)+g.sum(1)).clamp_min(1e-8)
    iou=pairwise_box_iou(boxes[q],gt[t]).diag()
    errors=((boxes[q,:3]-gt[t,:3])*shape).norm(dim=1)
    result=dict(nodules=len(gt),scans=1,selected=len(selected),coverage=0,
        coarse_dice=float(dice.sum()),box_iou=float(iou.sum()),center_error_mm=float(errors.sum()),
        true_positives=0,false_positives=0)
    if not len(selected):return result
    retained=match_coarse(objects[selected],boxes[selected],masks[selected],gt,batch['target_masks'])
    for qi,ti in zip(selected[retained.query_indices].tolist(),retained.target_indices.tolist()):
        center=(boxes[qi,:3]*shape-.5).cpu()
        origin=torch.floor(center+.5).long()-torch.tensor(roi_shape)//2
        crop=batch['target_mask_crops'][ti]
        visible=paste_compact_mask(crop,batch['target_mask_origins'][ti],origin,roi_shape).sum()
        result['coverage']+=int(visible>=minimum_coverage*crop.sum())
    radius=(gt[:,3:]*shape).amax(-1).mul(.5).clamp_min(3)
    distance=torch.cdist(boxes[selected,:3]*shape,gt[:,:3]*shape)/radius[None]
    cost=torch.where(distance<=1,distance,1e6)
    a,b=linear_sum_assignment(cost.cpu().numpy())
    result['true_positives']=int((distance[a,b]<=1).sum())
    near_gt=(distance<=1).any(1)
    ignore_count=0
    ignored_boxes=batch.get('ignored_target_boxes')
    if ignored_boxes is not None and len(ignored_boxes):
        ib=ignored_boxes.to(boxes)
        ir=(ib[:,3:]*shape).amax(-1).mul(.5).clamp_min(3)
        near_ignored=(torch.cdist(boxes[selected,:3]*shape,ib[:,:3]*shape)/ir[None]<=1).any(1)
        ignore_count=int((near_ignored&~near_gt).sum())
    result['false_positives']=len(selected)-result['true_positives']-ignore_count
    return result


def summarize_detection_counts(totals):
    n=max(totals['nodules'],1);scans=max(totals['scans'],1)
    return dict(nodules=int(totals['nodules']),scans=int(totals['scans']),
        coverage=totals['coverage']/n,coarse_dice=totals['coarse_dice']/n,
        box_iou=totals['box_iou']/n,mean_center_error_mm=totals['center_error_mm']/n,
        detection_recall=totals['true_positives']/n,
        false_positives_per_scan=totals['false_positives']/scans,
        selected_queries_per_scan=totals['selected']/scans)
