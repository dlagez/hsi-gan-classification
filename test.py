import torch
import torch.nn as nn

x = torch.tensor([[1.0, -1.0],
                  [0.0,  1.0],
                  [0.0,  0.0]])

in_features = x.shape[1]  # = 2
out_features = 2

m = nn.Linear(in_features, out_features)

y = m(x)


out_features = 9
m2 = nn.Linear(in_features, out_features)
m2.bias
m2.weight
y2 = m2(x)
