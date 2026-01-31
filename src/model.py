import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, num_nodes, input_dim=3, output_dim=1):
        super(TrafficGNN, self).__init__()
        # Two spatial layers to capture neighborhood patterns
        self.conv1 = GCNConv(input_dim, 32) 
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, output_dim)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, 3]
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.fc(x) # Returns a single tensor, not a tuple