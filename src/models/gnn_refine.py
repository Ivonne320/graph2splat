import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from src.modules.norm import ChannelLayerNorm32, GroupNorm32

def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    elif norm_type == "batch":
        return nn.BatchNorm3d(*args, **kwargs)

class SceneGraphRefiner(torch.nn.Module):
    # def __init__(self, embed_dim, heads=4):
    #     super().__init__()
    #     self.gnn1 = GATConv(embed_dim, embed_dim, heads=heads, concat=False)
    #     self.gnn2 = GATConv(embed_dim, embed_dim, heads=heads, concat=False)

    # def forward(self, z, edge_index):
    #     # z: (N_obj, embed_dim), edge_index: (2, num_edges)
    #     z = F.relu(self.gnn1(z, edge_index))
    #     z = self.gnn2(z, edge_index)
    #     return z
    
    def __init__(self, input_dim=9, hidden_dim=8, output_dim=9, heads=4, vol_shape=(64, 64, 64)):
        super().__init__()
        self.proj_scale = nn.Parameter(torch.tensor(0.5))
        self.gnn1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gnn2 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False)
        self.layernorm = nn.LayerNorm(output_dim)

        self.vol_shape = vol_shape
        self.feat_dim = output_dim
        
        self.projector = nn.Sequential(
            nn.Linear(output_dim, 32*4*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4, 4)), 

            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),          
            nn.ConvTranspose3d(16, output_dim, kernel_size=4, stride=2, padding=1),  
            norm_layer("batch", output_dim)
        )

    def forward(self, obj_feats, edge_index, volumes):
        """
        Args:
            obj_feats: (N, input_dim) pooled features from each decoded object
            edge_index: (2, num_edges) scene graph structure
            volumes: (N, C, D, H, W) decoded volumes from object embeddings
        Returns:
            fused_volumes: (N, C, D, H, W)
        """
        # GNN refinement
        x = F.relu(self.gnn1(obj_feats, edge_index))
        x = self.gnn2(x, edge_index)
        x = self.layernorm(x) + obj_feats
        
        proj = self.projector(x)    
 
        if proj.shape[-3:] != volumes.shape[-3:]:
            proj = F.interpolate(proj, size=volumes.shape[-3:], mode='trilinear', align_corners=False)

        return volumes + self.proj_scale*proj
        # return proj
