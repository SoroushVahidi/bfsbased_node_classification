"""
Diffusion-Jump GNN (DJ) architecture.

Vendored from: https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs
License: follow upstream repository (see docs/DJGNN_INTEGRATION.md).
"""
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, GCNConv
from .pump import *  # noqa: F403

'''
    Unsupervised Step of Diffusion-Jump:
    Arguments:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        out_channels: Number of output features
        adj_dim: Number of dimensions of the adjacency matrix
        num_centers: Number of columns of the eigenvectors of the graph Laplacian
    Returns:
        s: Projected eigenvectors of the graph Laplacian
        pump_loss: Pump loss
        distance_matrix: Matrix of distances between the projected eigenvectors of the graph Laplacian
'''

class DJ_unsupervised(torch.nn.Module):
    def __init__(self, adj_dim, num_centers):
        super(DJ_unsupervised, self).__init__()
        # Upstream reset torch.manual_seed(1234) here; removed so callers control RNG.
        self.MLP_adj = Linear(adj_dim, num_centers)
    def forward(self, adj):
        s = self.MLP_adj(adj)
        _, pump_loss, ortho_loss, distance_matrix= pump(adj, s)
        return s, pump_loss + ortho_loss, distance_matrix
'''
    Supervised Step of DJ: 
    Arguments:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        out_channels: Number of output features
        n_jumps: Number of jumps
    Returns:
        z: Output of the model
        
    
'''
class DJ_supervised(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, n_jumps,out_channels,drop_out = 0.5):
        super(DJ_supervised, self).__init__()
        cached = False
        add_self_loops = True
        save_mem  = False
        self.convs = torch.nn.ModuleList()
        self.bn = torch.nn.ModuleList()
        for i in range(n_jumps):
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))
        #Extra conv
        self.conv_extra = GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops)
        self.bn_extra = torch.nn.BatchNorm1d(hidden_channels)
        self.conv_extra2 = GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem , add_self_loops=add_self_loops)       
        self.bn_extra_2 = torch.nn.BatchNorm1d(hidden_channels)
        #Aux
        self.new_adj = None
        self.embeddings = None
        self.gnn_out = None
        self.drop_out = drop_out
        # Parameters
        self.att = torch.nn.Parameter(torch.ones(n_jumps + 1))
        self.sm = torch.nn.Softmax(dim=0)
        # Linear
        self.classify1 = Linear(hidden_channels*(n_jumps + 1), out_channels)
        # Adjs
    def forward(self, x, adj):
        mask_attentions = self.sm(self.att)
        extra_conv = self.conv_extra(x, adj[0])
        extra_conv = self.bn_extra(extra_conv)
        extra_conv = extra_conv.relu()
        extra_conv = F.dropout(extra_conv, p=0.5, training=self.training)
        extra_conv = self.conv_extra2(extra_conv, adj[0])
        extra_conv = self.bn_extra_2(extra_conv)
        extra_conv = extra_conv.relu() * mask_attentions[-1]     
        z_s = []
        for i, conv in enumerate(self.convs):
            z = conv(x, adj[i + 1]).relu() * mask_attentions[i]
            z_s.append(z)            
        final_z = torch.cat(z_s, dim=1)
        final_z = torch.cat([final_z, extra_conv], dim=1)
        final_z = F.dropout(final_z, p=self.drop_out, training=self.training)
        z = self.classify1(final_z).log_softmax(dim=-1)
        return z, torch.zeros(1).mean()
'''
    Diffusion-Jump:
    Arguments:
        in_channels: Number of input features
        hidden_channels: Number of hidden features
        out_channels: Number of output features
        adj_dim: Number of dimensions of the adjacency matrix
        num_centers: Number of columns of the eigenvectors of the graph Laplacian
        n_jumps: Number of jumps
    Returns:
        z: Output of the model
        loss = Pump loss + Orthogonality loss
        
        
'''
class DJ(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_centers,adj_dim, n_jumps,out_channels,drop_out = 0.5):
        super(DJ, self).__init__()
        self.MLP = Linear(adj_dim, num_centers)
        #GNN
        self.convs = torch.nn.ModuleList()
        for i in range(n_jumps):
            self.convs.append(DenseGCNConv(in_channels, hidden_channels))
        #Extra conv
        self.conv_extra = DenseGCNConv(in_channels, hidden_channels)
        self.conv_extra2 = DenseGCNConv(hidden_channels, hidden_channels)
        #Aux
        self.drop_out = drop_out
        self.new_adj = None
        self.embeddings = None
        self.gnn_out = None
        # Parameters
        self.att = torch.nn.Parameter(torch.ones(n_jumps + 1))
        self.sm = torch.nn.Softmax(dim=0)
        # Linear
        self.classify1 = Linear(hidden_channels*(n_jumps + 1), out_channels)
    def forward(self, x, adj):
        mask_attentions = self.sm(self.att)
        extra_conv = self.conv_extra(x, adj.unsqueeze(0)).squeeze(0).relu() 
        extra_conv = F.dropout(extra_conv, p=self.drop_out, training=self.training)
        extra_conv = self.conv_extra2(extra_conv, adj.unsqueeze(0)).squeeze(0).relu() * mask_attentions[-1]
        s = self.MLP(adj)
        _, pump_loss, ortho_loss,distance_matrix= pump(adj, s)
        distance_matrix = distance_matrix.squeeze(0)
        new_adj = distance_matrix
        self.embeddings = s
        z_s = []
        previous_adj = torch.zeros_like(new_adj)
        for i, conv in enumerate(self.convs):
            adj = torch.zeros_like(new_adj)
            top_min = torch.topk(new_adj, i, dim=1, largest=False, sorted=True)
            adj.scatter_(1, top_min.indices, 1)
            adj = torch.mul(adj, torch.exp(-new_adj))
            adj = adj - previous_adj
            z = conv(x, adj.unsqueeze(0)).squeeze(0).relu() * mask_attentions[i]
            z_s.append(z)
            previous_adj = adj
        del previous_adj
        del adj
        final_z = torch.cat(z_s, dim=1)
        final_z = torch.cat([final_z, extra_conv], dim=1)
        final_z = F.dropout(final_z, p=self.drop_out, training=self.training)
        z = self.classify1(final_z).log_softmax(dim=-1)    
        self.gnn_out = z.detach().cpu().numpy()
        return z, (pump_loss + ortho_loss)
