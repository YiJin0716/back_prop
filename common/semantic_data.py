"""Same physical-nodule policy as v2, without a malignancy semantic target."""
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from back_prop.common.coarse_data import (
    WholeCTLIDCDataset as V2Dataset, DEFAULT_MANIFEST, load_cases, whole_ct_collate,
)
from back_prop.common.features import SEMANTIC_INDICES_V2


class WholeCTLIDCDataset(V2Dataset):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        for field in ('semantic_histograms', 'semantic_targets', 'semantic_reader_counts'):
            batch[field] = batch[field][:, list(SEMANTIC_INDICES_V2)]
        # The segmentation input is windowed to [-1024,1024]. Radiomics must
        # instead use the original HU values on the identical 1-mm grid.
        image = nib.as_closest_canonical(nib.load(self.cases[index]['image']))
        zoom = nib.affines.voxel_sizes(image.affine) / self.target_spacing
        hu = ndimage.zoom(np.asarray(image.dataobj, dtype=np.float32), zoom,
                          order=1, mode='nearest', prefilter=False)
        batch['image_hu'] = torch.from_numpy(hu).unsqueeze(0)
        if batch['image_hu'].shape != batch['image'].shape:
            raise ValueError('Original-HU and network-input grids differ')
        return batch
