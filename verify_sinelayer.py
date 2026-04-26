import torch
import torch.nn as nn
import math
from pinn import SineLayer

# Test 1: output shape
layer = SineLayer(in_features=2, out_features=64, is_first=True, omega_0=30)
x = torch.randn(10, 2)
out = layer(x)
assert out.shape == (10, 64), f"shape wrong: {out.shape}"

# Test 2: first-layer weight bounds
w = layer.linear.weight.data
bound = 1.0 / 2  # 1/in_features
assert w.abs().max() <= bound + 1e-6, f"first-layer weight out of bounds: {w.abs().max()}"
b = layer.linear.bias.data
assert b.abs().max() <= bound + 1e-6, f"first-layer bias out of bounds: {b.abs().max()}"

# Test 3: hidden-layer weight bounds
layer_h = SineLayer(in_features=64, out_features=64, is_first=False, omega_0=30)
w_h = layer_h.linear.weight.data
bound_h = math.sqrt(6.0 / 64) / 30
assert w_h.abs().max() <= bound_h + 1e-6, f"hidden-layer weight out of bounds: {w_h.abs().max()}"
b_h = layer_h.linear.bias.data
assert b_h.abs().max() <= bound_h + 1e-6, f"hidden-layer bias out of bounds: {b_h.abs().max()}"

# Test 4: output is sin(omega_0 * Wx + b)
layer2 = SineLayer(in_features=2, out_features=4, is_first=True, omega_0=30)
x2 = torch.tensor([[1.0, 2.0]])
expected = torch.sin(30.0 * layer2.linear(x2))
out2 = layer2(x2)
assert torch.allclose(out2, expected), "forward not sin(omega_0 * linear(x))"

print("SineLayer: все тесты прошли")
