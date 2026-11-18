# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import dataclasses

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestParallelMLP:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.mlp = MLP(transformer_config, get_gpt_layer_local_submodules().mlp.submodules)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.mlp, MLP)

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == 1212

    """
    def test_cpu_forward(self, mlp):
        # [sequence length, micro batch size, hidden size]
        hidden_states = torch.ones((32, 2, mlp.config.hidden_size))
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == torch.float32
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        mlp = self.mlp
        mlp.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        assert output_bias.device.type == 'cuda'


class _ScaledActivation(torch.nn.Module):
    """An activation module with a learnable parameter, like Apertus' xIELU."""

    def __init__(self, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        return self.scale * x


class TestMLPActivationModule:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            use_te_activation_func=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _submodules(self, activation_func):
        submodules = get_gpt_layer_local_submodules().mlp.submodules
        return dataclasses.replace(submodules, activation_func=activation_func)

    def test_module_activation_is_used_without_te_activation_func(self):
        mlp = MLP(self.config, self._submodules(_ScaledActivation))
        assert isinstance(mlp.activation_func, _ScaledActivation)
        # Its parameter is part of the MLP, so it is trained and checkpointed.
        assert "activation_func.scale" in dict(mlp.named_parameters())

    def test_te_activation_op_needs_te_activation_func(self):
        from megatron.core.extensions.transformer_engine import TEActivationOp

        if TEActivationOp is None:
            pytest.skip("Transformer Engine >= 1.13 not available")
        # The MoE spec passes TEActivationOp to shared experts whatever the config says.
        mlp = MLP(self.config, self._submodules(TEActivationOp))
        assert mlp.activation_func is self.config.activation_func
