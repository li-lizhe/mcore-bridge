# Copyright (c) ModelScope Contributors. All rights reserved.
"""The GDN sequence-parallel toggle has to reach the projection parameters.

``GatedDeltaNet`` gathers the sequence before the projections and turns sequence parallelism off on them for
that call, because Transformer Engine then all-reduces the projection gradient itself. Megatron's gradient
finalizer decides from ``param.sequence_parallel`` (see ``finalize_model_grads``) whether a gradient still
has to be summed over the tensor parallel group, so the flag has to be off on the parameters as well: with
only the module attribute off, the finalizer sums the already complete gamma gradient once more and
multiplies it by the tensor parallel size. Parameters that do not carry the flag are left alone.
"""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace


class _FakeLayerNormLinear(nn.Module):
    """Stand-in for a Transformer Engine ``LayerNormLinear``.

    Transformer Engine flags the module and the parameters it wants reduced over the tensor parallel group
    with ``sequence_parallel``; ``layer_norm_weight`` is the one the reported case loses in the finalizer.
    """

    def __init__(self, sequence_parallel: bool = True, mark_parameters: bool = True):
        super().__init__()
        self.sequence_parallel = sequence_parallel
        self.layer_norm_weight = nn.Parameter(torch.ones(4))
        self.weight = nn.Parameter(torch.ones(2, 4))
        if mark_parameters:
            self.layer_norm_weight.sequence_parallel = sequence_parallel


def _gdn():
    """Return ``GatedDeltaNet``, skipping the module when Megatron is not installed."""
    pytest.importorskip('megatron.core')
    from mcore_bridge.model.modules.gated_delta_net import GatedDeltaNet
    return GatedDeltaNet


def _projections(**modules):
    """Only the four projection attributes the toggle walks are needed."""
    attributes = {'in_proj': None, 'in_proj_qkvz': None, 'in_proj_ba': None, 'out_proj': None}
    attributes.update(modules)
    return SimpleNamespace(**attributes)


def _finalize_layernorm_grad(param, tp_size):
    """Megatron's branch: a sequence-parallel parameter has its complete gradient summed over the TP group."""
    if getattr(param, 'sequence_parallel', False):
        return param.main_grad * tp_size
    return param.main_grad


def test_toggle_marks_the_parameters_and_the_restore_keeps_them_off():
    gated_delta_net = _gdn()
    layer = _FakeLayerNormLinear()
    model = _projections(in_proj=layer)
    saved = gated_delta_net._set_linear_sequence_parallel(model, False)

    assert layer.sequence_parallel is False
    assert layer.layer_norm_weight.sequence_parallel is False
    assert len(saved) == 2
    assert saved[0][0] is layer
    assert saved[1][0] is layer.layer_norm_weight

    gated_delta_net._restore_linear_sequence_parallel(model, saved)
    # The module goes back to its configured value, the parameter stays off: the finalizer reads it after
    # the backward, long after the module attribute was restored.
    assert layer.sequence_parallel is True
    assert layer.layer_norm_weight.sequence_parallel is False


def test_the_finalizer_leaves_the_gamma_gradient_unscaled():
    gated_delta_net = _gdn()
    layer = _FakeLayerNormLinear()
    model = _projections(in_proj=layer)
    saved = gated_delta_net._set_linear_sequence_parallel(model, False)
    gated_delta_net._restore_linear_sequence_parallel(model, saved)

    layer.layer_norm_weight.main_grad = torch.ones(4)
    assert torch.equal(_finalize_layernorm_grad(layer.layer_norm_weight, 4), torch.ones(4))


def test_parameters_without_the_flag_are_left_alone():
    gated_delta_net = _gdn()
    layer = _FakeLayerNormLinear(mark_parameters=False)
    model = _projections(in_proj=layer)
    saved = gated_delta_net._set_linear_sequence_parallel(model, False)

    assert not hasattr(layer.weight, 'sequence_parallel')
    assert len(saved) == 1
    assert saved[0][0] is layer

    gated_delta_net._restore_linear_sequence_parallel(model, saved)
    assert layer.sequence_parallel is True
