"""Mask-guided geometry and confidence for the V5 query detector.

The detector still predicts one coarse mask per learned query. Its box starts
from that mask's spatial moments, and its confidence sees mask-pooled local
features. Both are computed from predictions alone, in training and inference.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from back_prop.common.base_model import MLP
from back_prop.common.geometry import V3MaskDETR


@torch.no_grad()
def select_mask_queries(object_logits, mask_logits, threshold=.5, overlap_threshold=.5):
    """Suppress duplicate predicted instances using mask IoU, without labels."""
    candidates=(object_logits.sigmoid()>=threshold).nonzero().flatten()
    if len(candidates)<2:return candidates
    foreground=(mask_logits[candidates].float().sigmoid()>=.5).flatten(1).float()
    intersection=foreground@foreground.T
    mass=foreground.sum(1)
    iou=intersection/(mass[:,None]+mass[None,:]-intersection).clamp_min(1)
    order=object_logits[candidates].argsort(descending=True,stable=True).tolist()
    selected=[]
    while order:
        first=order.pop(0);selected.append(first)
        order=[i for i in order if iou[first,i]<=overlap_threshold]
    return candidates[torch.tensor(selected,device=candidates.device)]


def mask_geometry(mask_logits, features, power=4.):
    """Differentiable normalized centers, extents, local features and statistics.

    Power weighting suppresses the large volume of low-probability background.
    Log-space normalization also handles empty predictions without NaNs. For a
    filled ellipsoid, sqrt(20 * coordinate variance) estimates its full extent.
    """
    shape = mask_logits.shape[-3:]
    with torch.autocast(mask_logits.device.type, enabled=False):
        logits = mask_logits.float().flatten(2)
        probability = logits.sigmoid()
        weight = (F.logsigmoid(logits)*power).softmax(-1)
        coordinates = torch.stack(torch.meshgrid(*(
            (torch.arange(n,device=logits.device,dtype=torch.float32)+.5)/n for n in shape
        ),indexing='ij')).flatten(1)
        center = weight@coordinates.T
        variance = (weight@coordinates.square().T-center.square()).clamp_min(0)
        minimum = logits.new_tensor([1/n for n in shape])
        extent = (20*variance+1e-8).sqrt().clamp(max=1.)
        extent = torch.maximum(extent,minimum)
        pooled = weight@features.float().flatten(2).transpose(1,2)
        statistics = torch.cat((probability.sum(-1,keepdim=True).log1p()/10,
            probability.amax(-1,keepdim=True),(weight*probability).sum(-1,keepdim=True),
            center,extent),dim=-1)
    return center,extent,pooled,statistics


class MaskGuidedDETR(V3MaskDETR):
    """Keep DETR masks; ground localization and confidence in each mask."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden = self.query_embedding.shape[-1]
        mask_dim = self.mask_feature_projection[0].out_channels
        del self.object_head, self.box_head
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,64),nn.GELU())
        self.local_projection = nn.Sequential(nn.LayerNorm(mask_dim),nn.Linear(mask_dim,64),nn.GELU())
        self.prediction_head = MLP(128+9,128,7,3)
        nn.init.zeros_(self.prediction_head.layers[-1].weight)
        nn.init.zeros_(self.prediction_head.layers[-1].bias)
        with torch.no_grad():self.prediction_head.layers[-1].bias[0]=-2.4

    @classmethod
    def from_legacy(cls, legacy):
        model = cls(num_queries=legacy.num_queries,hidden_dim=legacy.query_embedding.shape[-1],
            coarse_shape=legacy.coarse_shape,nheads=legacy.decoder.layers[0].self_attn.num_heads,
            encoder_layers=len(legacy.encoder.layers),decoder_layers=len(legacy.decoder.layers),
            input_channels=legacy.input_projection[0].in_channels,
            mask_dim=legacy.mask_feature_projection[0].out_channels)
        shared = {k:v for k,v in legacy.state_dict().items() if not k.startswith(('object_head.','box_head.'))}
        incompatible = model.load_state_dict(shared,strict=False)
        if incompatible.unexpected_keys or any(not k.startswith(('query_projection.','local_projection.','prediction_head.'))
                                              for k in incompatible.missing_keys):
            raise ValueError(f'Detector transfer mismatch: {incompatible}')
        return model

    def _forward(self, screening):
        coarse = F.adaptive_avg_pool3d(screening,self.coarse_shape)
        encoded_grid = self.input_projection(coarse)
        features = encoded_grid.flatten(2).transpose(1,2)
        memory = self.encoder(features+self.position_embedding)
        queries = self.query_embedding.expand(screening.shape[0],-1,-1)
        decoded = self.decoder(queries,memory)
        mask_features = self.mask_feature_projection(screening)
        context = F.interpolate(self.context_projection(encoded_grid).float(),size=screening.shape[-3:],
                                mode='trilinear',align_corners=False)
        mask_features = mask_features+context
        embeddings = self.mask_embedding(decoded)
        mask_logits = torch.einsum('bqc,bcdhw->bqdhw',embeddings,mask_features).float()
        center,extent,local,statistics = mask_geometry(mask_logits,mask_features)
        prediction = self.prediction_head(torch.cat((self.query_projection(decoded),
            self.local_projection(local),statistics),dim=-1)).float()
        object_logits = prediction[...,0]
        centers = (center+.5*extent*prediction[...,1:4].tanh()).clamp(0,1)
        sizes = (extent*torch.exp(math.log(2)*prediction[...,4:7].tanh())).clamp(max=1.)
        return object_logits,torch.cat((centers,sizes),dim=-1),mask_logits,decoded


class NativeSeededDETR(MaskGuidedDETR):
    """Use native VISTA instances as query priors, keeping unseeded queries.

    Native centers avoid coarse-grid quantization. Trainable local heads adjust
    confidence, boxes and masks; every query still participates in set matching.
    No annotations are used to construct or order the priors.
    """
    def __init__(self,confidence_residual_bound=None,**kwargs):
        super().__init__(**kwargs)
        if confidence_residual_bound is not None and (
                not math.isfinite(confidence_residual_bound) or confidence_residual_bound<=0):
            raise ValueError('Native confidence residual bound must be positive')
        self.confidence_residual_bound=confidence_residual_bound
        hidden=self.query_embedding.shape[-1]
        channels=self.mask_feature_projection[0].out_channels
        self.native_query=MLP(channels+9,hidden,hidden,2)
        self.native_prediction=nn.Linear(hidden,7)
        self.native_mask_embedding=nn.Linear(hidden,channels)
        for layer in (self.native_prediction,self.native_mask_embedding):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    @classmethod
    def from_mask_guided(cls,model,confidence_residual_bound=None):
        result=cls(num_queries=model.num_queries,hidden_dim=model.query_embedding.shape[-1],
            coarse_shape=model.coarse_shape,nheads=model.decoder.layers[0].self_attn.num_heads,
            encoder_layers=len(model.encoder.layers),decoder_layers=len(model.decoder.layers),
            input_channels=model.input_projection[0].in_channels,
            mask_dim=model.mask_feature_projection[0].out_channels,
            confidence_residual_bound=confidence_residual_bound)
        incompatible=result.load_state_dict(model.state_dict(),strict=False)
        if incompatible.unexpected_keys or any(not key.startswith('native_') for key in incompatible.missing_keys):
            raise ValueError(f'Native detector transfer mismatch: {incompatible}')
        return result

    def forward(self,screening,proposals=None):
        if proposals is None:
            raise ValueError('Native-seeded DETR requires prediction-derived native proposals')
        if screening.shape[0]!=1:
            raise ValueError('Native proposals belong to a single whole CT')
        objects,boxes,masks,queries=super().forward(screening)
        count=self.seeded_count(proposals)
        if not count:return objects,boxes,masks,queries
        # Reuse the learned spatial representation. These inexpensive projected
        # maps are recomputed to keep the existing DETR checkpoint path intact.
        encoded=self.input_projection(F.adaptive_avg_pool3d(screening,self.coarse_shape))
        features=self.mask_feature_projection(screening)+F.interpolate(
            self.context_projection(encoded).float(),size=screening.shape[-3:],
            mode='trilinear',align_corners=False)
        prior=proposals.coarse_logits[:count].to(screening.device).float()
        seed_boxes=proposals.boxes[:count].to(screening.device).float()
        _,_,local,statistics=mask_geometry(prior[None],features)
        statistics=torch.cat((statistics[0,:,:3],seed_boxes),dim=-1)
        query=self.native_query(torch.cat((local[0],statistics),dim=-1))
        prediction=self.native_prediction(query).float()
        correction=prediction[:,0]
        if self.confidence_residual_bound is not None:
            bound=self.confidence_residual_bound
            # Coarse local features may lose small-nodule evidence retained by
            # the native VISTA prior. Learn a bounded log-odds correction rather
            # than allowing that coarse head to erase arbitrarily strong priors.
            correction=bound*(correction/bound).tanh()
        confidence=proposals.object_logits[:count].to(screening.device).float()+correction
        center=(seed_boxes[:,:3]+.5*seed_boxes[:,3:]*prediction[:,1:4].tanh()).clamp(0,1)
        extent=(seed_boxes[:,3:]*torch.exp(math.log(2)*prediction[:,4:7].tanh())).clamp(max=1.)
        residual=torch.einsum('qc,bcdhw->bqdhw',self.native_mask_embedding(query),features).float()
        seeded_masks=prior[None]+2*residual.tanh()
        return (torch.cat((confidence[None],objects[:,count:]),dim=1),
            torch.cat((torch.cat((center,extent),dim=-1)[None],boxes[:,count:]),dim=1),
            torch.cat((seeded_masks,masks[:,count:]),dim=1),
            torch.cat((query[None],queries[:,count:]),dim=1))

    def seeded_count(self,proposals):
        # Retain global search capacity even when a scan contains many native
        # components. Otherwise a missed nodule could have no movable query.
        reserved=max(1,self.num_queries//6)
        return min(len(proposals.boxes),max(1,self.num_queries-reserved))
