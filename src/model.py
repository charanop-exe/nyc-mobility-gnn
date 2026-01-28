import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim):
        super(TrafficGNN, self).__init__()
        # 1. Spatial Layer: Learns how neighborhoods affect each other
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 8)
        
        # 2. Fully Connected Layer: Produces the final demand prediction
        self.fc = torch.nn.Linear(8, output_dim)

    def forward(self, x, edge_index):
        # x: Demand data (Nodes)
        # edge_index: Connectivity (Adjacency Matrix)
        
        # Apply Spatial Convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Activation function
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Predict the next time step
        x = self.fc(x)
        return x