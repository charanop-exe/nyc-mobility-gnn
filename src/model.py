# src/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TrafficGNN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=2, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index):
        # x_seq: [B, T, N, F]
        B, T, N, Fin = x_seq.shape

        temporal_features = []

        for t in range(T):
            xt = x_seq[:, t].reshape(-1, Fin)          # [B*N, F]
            h = F.relu(self.gat1(xt, edge_index))
            h = F.relu(self.gat2(h, edge_index))
            h = h.reshape(B, N, -1)                    # [B, N, H]
            temporal_features.append(h)

        h_seq = torch.stack(temporal_features, dim=1)  # [B, T, N, H]

        # Apply GRU **per zone**
        h_seq = h_seq.permute(0, 2, 1, 3)               # [B, N, T, H]
        h_seq = h_seq.reshape(B * N, T, -1)             # [B*N, T, H]

        _, h_final = self.gru(h_seq)                    # [1, B*N, H]
        out = self.fc(h_final.squeeze(0))               # [B*N, 1]

        return out.view(B, N)                            # ✅ [B, N]
