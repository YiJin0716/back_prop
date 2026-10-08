import torch

from back_prop.common.refiner import FineMaskRefiner3D
from back_prop.common.roi import (
    assemble_refiner_input,
    integer_crop,
    paste_compact_mask,
    sample_global_at_roi,
)


def test_integer_crop_preserves_voxels_and_reports_origin():
    volume = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(1, 5, 6, 7)
    patch, origin, valid = integer_crop(volume, (2, 3, 3), (3, 3, 3))
    assert origin.tolist() == [1, 2, 2]
    assert torch.equal(patch, volume[:, 1:4, 2:5, 2:5])
    assert valid.shape == (1, 3, 3, 3)
    assert valid.all()


def test_integer_crop_boundary_padding_and_value_gradient():
    volume = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    volume.requires_grad_(True)
    patch, origin, valid = integer_crop(volume, (0, 0, 0), (3, 3, 3), pad_value=-1.0)
    assert origin.tolist() == [-1, -1, -1]
    assert torch.equal(patch[:, 1:, 1:, 1:], volume[:, :2, :2, :2])
    assert torch.all(patch[:, 0] == -1)
    assert torch.all(patch[:, :, 0] == -1)
    assert torch.all(patch[:, :, :, 0] == -1)
    assert valid.sum().item() == 8
    patch.sum().backward()
    assert volume.grad is not None
    assert volume.grad[:, :2, :2, :2].eq(1).all()
    assert volume.grad.sum().item() == 8


def test_global_sampling_uses_original_voxel_centres_and_has_gradient():
    source = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 1, 4, 5, 6)
    source.requires_grad_(True)
    sampled = sample_global_at_roi(source, origin=(-1, 1, 2), scan_shape=(4, 5, 6), size=(3, 3, 3))
    expected, _, _ = integer_crop(source, center_xyz=(0, 2, 3), size=(3, 3, 3))
    assert sampled.shape == (1, 1, 3, 3, 3)
    assert torch.allclose(sampled, expected, atol=1e-5)
    sampled.sum().backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert source.grad.abs().sum() > 0


def test_paste_compact_mask_uses_full_scan_origins():
    compact = torch.ones((2, 2, 2), dtype=torch.uint8)
    target = paste_compact_mask(
        compact,
        mask_origin=(2, 3, 4),
        roi_origin=(1, 2, 3),
        size=(4, 4, 4),
    )
    expected = torch.zeros((4, 4, 4), dtype=torch.uint8)
    expected[1:3, 1:3, 1:3] = 1
    assert torch.equal(target, expected)


def test_refiner_input_contract_and_zero_initialized_delta():
    batch, shape = 2, (9, 10, 11)
    ct = torch.randn(batch, 1, *shape)
    vista = torch.rand(batch, 1, *shape)
    coarse_logits = torch.randn(batch, 1, *shape)
    valid = torch.ones(batch, 1, *shape, dtype=torch.bool)
    context = torch.randn(batch, 8, *shape)
    inputs = assemble_refiner_input(ct, vista, coarse_logits, valid, context)
    assert inputs.shape == (batch, 12, *shape)

    model = FineMaskRefiner3D(base_channels=2, use_checkpoint=False)
    query = torch.randn(batch, 128)
    delta = model(inputs, query)
    assert delta.shape == (batch, 1, *shape)
    assert torch.count_nonzero(delta) == 0
    assert torch.count_nonzero(model.residual_head.weight) == 0
    for module in model.modules():
        if hasattr(module, "film"):
            assert torch.count_nonzero(module.film.weight) == 0
            assert torch.count_nonzero(module.film.bias) == 0


def test_trained_refiner_path_propagates_to_input_and_query():
    # Exact zero initialization intentionally makes the initial delta constant,
    # so query/input gradients begin after the zero heads receive an update.
    # Simulate that post-update state to verify the complete autograd path.
    model = FineMaskRefiner3D(input_channels=4, query_dim=8, base_channels=2, use_checkpoint=True)
    with torch.no_grad():
        model.residual_head.weight.fill_(1e-2)
        model.encoder_1.film.weight.fill_(1e-2)
    model.train()
    value = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    query = torch.randn(1, 8, requires_grad=True)
    output = model(value, query)
    output.square().mean().backward()
    assert output.shape == (1, 1, 8, 8, 8)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert value.grad.abs().sum() > 0
    assert query.grad.abs().sum() > 0
