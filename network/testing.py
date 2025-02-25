import torch
from network.shape_prior import RefinePrior, CrossAttention, Attention, Mlp
from network.net_configs import get_example_config

config = get_example_config()

def test_refine_prior():
    model = RefinePrior(config)
    shape_tokens = torch.randn(2, config.n_classes, config.hidden_size)
    output = model(shape_tokens)
    assert output.shape == shape_tokens.shape, "RefinePrior output shape mismatch"
    print("RefinePrior test passed")

def test_cross_attention():
    model = CrossAttention(config)
    query_tokens = torch.randn(2, 5, config.hidden_size)
    key_value_tokens = torch.randn(2, 10, config.hidden_size)
    output = model(query_tokens, key_value_tokens)
    assert output.shape == (2, 5, config.hidden_size), "CrossAttention output shape mismatch"
    print("CrossAttention test passed")

def test_attention():
    model = Attention(config)
    hidden_states = torch.randn(2, config.n_classes, config.hidden_size)
    output = model(hidden_states)
    assert output.shape == (2, config.n_classes, config.hidden_size), "Attention output shape mismatch"
    print("Attention test passed")

def test_mlp():
    model = Mlp(config)
    inputs = torch.randn(2, 10, config.hidden_size)
    output = model(inputs)
    assert output.shape == (2, 10, config.hidden_size), "MLP output shape mismatch"
    print("MLP test passed")

if __name__ == "__main__":
    test_refine_prior()
    test_cross_attention()
    test_attention()
    test_mlp()
