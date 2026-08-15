import torch

from pointllm.model_cvpr.relation import SimpleRelationModule


def _grad_norm(parameter: torch.nn.Parameter) -> float:
    if parameter.grad is None:
        return 0.0
    return float(parameter.grad.abs().sum())


def test_non_adaln_identity_initialization_becomes_fully_trainable():
    torch.manual_seed(7)
    module = SimpleRelationModule(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_adaln=False,
    )
    module.train()
    layer = module.encoder.layers[0]

    x = torch.randn(2, 5, 16)
    target = torch.randn_like(x)
    initial = module(x)
    torch.testing.assert_close(initial, x, rtol=0.0, atol=0.0)

    torch.nn.functional.mse_loss(initial, target).backward()
    assert _grad_norm(layer.mhsa.W_0.weight) > 0.0
    assert _grad_norm(layer.linear[3].weight) > 0.0
    assert _grad_norm(layer.mhsa.to_qvk.weight) == 0.0
    assert _grad_norm(layer.linear[0].weight) == 0.0

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    updated = module(x)
    assert not torch.equal(updated, x)
    torch.nn.functional.mse_loss(updated, target).backward()
    assert _grad_norm(layer.mhsa.to_qvk.weight) > 0.0
    assert _grad_norm(layer.linear[0].weight) > 0.0
    q_grad, k_grad, v_grad = layer.mhsa.to_qvk.weight.grad.chunk(3, dim=0)
    assert float(q_grad.abs().sum()) > 0.0
    assert float(k_grad.abs().sum()) > 0.0
    assert float(v_grad.abs().sum()) > 0.0

    with torch.no_grad():
        original_residual = module(x) - x
        perturbed = x.clone()
        # A uniform shift is removed by pre-norm LayerNorm, so perturb channels.
        perturbed[:, :2] += torch.linspace(-0.5, 0.5, x.size(-1))
        perturbed_residual = module(perturbed) - perturbed
    assert not torch.allclose(
        original_residual[:, 2:], perturbed_residual[:, 2:], rtol=1e-5, atol=1e-7
    )


def test_non_adaln_reinitialization_restores_trainable_identity():
    module = SimpleRelationModule(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_adaln=False,
    )
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1.0)

    module.reset_parameters_for_training()
    layer = module.encoder.layers[0]
    x = torch.randn(2, 5, 16)

    torch.testing.assert_close(module(x), x, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(layer.mhsa.to_qvk.weight) > 0
    assert torch.count_nonzero(layer.linear[0].weight) > 0
    assert torch.count_nonzero(layer.mhsa.W_0.weight) == 0
    assert torch.count_nonzero(layer.linear[3].weight) == 0


def test_adaln_zero_initialization_becomes_condition_dependent():
    torch.manual_seed(11)
    module = SimpleRelationModule(
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_adaln=True,
    )
    module.train()
    layer = module.encoder.layers[0]
    modulator = module.encoder.modulator[1]

    x = torch.randn(2, 5, 16)
    cond = torch.randn(2, 16)
    target = torch.randn_like(x)
    initial = module(x, cond=cond)
    torch.testing.assert_close(initial, x, rtol=0.0, atol=0.0)

    torch.nn.functional.mse_loss(initial, target).backward()
    assert _grad_norm(modulator.weight) > 0.0
    assert _grad_norm(layer.mhsa.to_qvk.weight) == 0.0

    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    updated = module(x, cond=cond)
    assert not torch.equal(updated, x)
    torch.nn.functional.mse_loss(updated, target).backward()
    assert _grad_norm(layer.mhsa.to_qvk.weight) > 0.0
    assert _grad_norm(layer.linear[0].weight) > 0.0

    with torch.no_grad():
        other_condition = module(x, cond=-cond)
    assert not torch.equal(updated, other_condition)


def test_attention_mask_uses_true_for_masked_positions():
    from pointllm.model_cvpr.relation import compute_mhsa

    q = torch.tensor([[[[1.0]]]])
    k = torch.tensor([[[[1.0], [1.0]]]])
    v = torch.tensor([[[[2.0], [100.0]]]])
    mask = torch.tensor([[[[False, True]]]])

    output = compute_mhsa(q, k, v, mask=mask)
    torch.testing.assert_close(output, torch.tensor([[[[2.0]]]]))
