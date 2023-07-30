import torch
import torch.nn.functional as F
from torch import nn
import numpy as np



class GaussianFourierFeatureTransform(torch.nn.Module):
    def __init__(self, num_input_channels, mapping_size=93, scale=25):
        super().__init__()
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale)
    def forward(self, x):
        x = x.squeeze(0)
        x = torch.sin(x @ self._B.to(x.device))
        return x

class Nerf_positional_embedding(torch.nn.Module):
    def __init__(self, multires,include_input):
        super().__init__()
        self.include_input = include_input
        self.periodic_fns = [torch.sin, torch.cos]
        self.brand_number = multires

    def forward(self, x):
        x = x.view(-1, 1)
        freq_bands = 2.**torch.linspace(0, self.brand_number-1, self.brand_number)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim, out_dim, activation = "relu"):
        self.activation = activation
        super().__init__(in_dim, out_dim,)
    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class No_positional_encoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.squeeze(0)
        return x

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        multries = 2
        self.embedder = Nerf_positional_embedding(multries, include_input = True)
        self.dense_layer_1 = DenseLayer(in_dim=2 , out_dim = 2*multries+1, activation="relu")
        self.dense_layer_2 = DenseLayer(in_dim=2 , out_dim = 2*multries+1, activation="relu")
        self.linear1 = nn.Linear((2*multries+1)*2, (2*multries+1)*4)
        self.linear2 = nn.Linear((2*multries+1)*4, 2)


    def forward(self, x):
        h1 = self.embedder(torch.unsqueeze(x[:,2], dim=1)).float()
        h2 = self.embedder(torch.unsqueeze(x[:,5], dim=1)).float()
        h3 = h1 + F.relu(self.dense_layer_1(torch.reshape(x[:,:2],(-1, 2)).float()))
        h4 = h2 + F.relu(self.dense_layer_2(torch.reshape(x[:,3:5],(-1, 2)).float()))
        # print("shape:", h1.shape, h3.shape)
        h = torch.cat([h3, h4], dim=1)
        # print("h.shape:", h.shape)
        x = F.relu(self.linear1(h))
        x = self.linear2(x)
        # x = F.softmax(x, dim=-1)
        return x

