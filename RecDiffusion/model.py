import math
import e3nn
import torch

from torch import nn
# , einsum
# from e3nn.nn.models.gate_points_2101 import Network

model_kwargs = {
    "irreps_in": None,  # no input features
    "irreps_hidden": e3nn.o3.Irreps("5x0e + 5x0o + 5x1e + 5x1o"),  # hyperparameter
    "irreps_out": "1x1o",  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": None,
    "irreps_edge_attr": e3nn.o3.Irreps.spherical_harmonics(3),
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = e3nn.nn.models.gate_points_2101.Network(
    **model_kwargs
)  # initializing model with parameters above


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings