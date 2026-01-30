import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, num_nodes, input_dim=3, output_dim=1):
        super(TrafficGNN, self).__init__()
        
        # 1. Spatial Layer 1: Processes the 3 features (Demand, Hour, Day)
        # We increase the hidden size to 32 to handle the extra temporal context
        self.conv1 = GCNConv(input_dim, 32) 
        
        # 2. Spatial Layer 2: Aggregates neighbor information
        self.conv2 = GCNConv(32, 16)
        
        # 3. Fully Connected Layer: Produces the final demand prediction
        self.fc = torch.nn.Linear(16, output_dim)

    def forward(self, x, edge_index):
        # x shape: [num_nodes, 3]
        # edge_index shape: [2, num_edges]
        
        # First Graph Convolution + ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        
        # Second Graph Convolution + ReLU activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Final prediction (Output dimension remains 1 for 'demand')
        x = self.fc(x)
        return x