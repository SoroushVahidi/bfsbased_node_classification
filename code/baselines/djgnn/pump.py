# "Pump" loss / orthogonality — vendored from:
# https://github.com/AhmedBegggaUA/Diffusion-Jump-GNNs (see docs/DJGNN_INTEGRATION.md)
import torch
# Trace of a tensor [1,k,k]
def _rank3_trace(x):
    return torch.einsum('ijj->i', x)

# Diagonal version of a tensor [1,n] -> [1,n,n]
def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1)) 
    return out

def pump(adj, s): 
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj # adj torch.Size([20, N, N]) N=Mmax
    s = s.unsqueeze(0) if s.dim() == 2 else s # s torch.Size([20, N, k])
    k = s.size(-1)
    s = torch.tanh(s) # torch.Size([20, N, k]) One k for each N of each graph
    d_flat = torch.einsum('ijk->ij', adj) # torch.Size([20, N]) 
    d = _rank3_diag(d_flat) # d torch.Size([20, N, N]) 
    
    CT_num = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2),adj), s)) # Tr(S^T A S) 
    CT_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))  # Tr(S^T D S) 
    adj = torch.cdist(s,s,p=2) # Distance matrix
    vol = _rank3_trace(d) # Vol(G)
    adj = (adj) / vol.unsqueeze(1).unsqueeze(1) # Distance matrix normalized
    # Mask with adjacency if proceeds 
    CT_loss = -(CT_num / CT_den) # Tr(S^T A S) / Tr(S^T D S)
    CT_loss = torch.mean(CT_loss)
    
    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)  
    i_s = torch.eye(k).type_as(ss) 
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s )  
    ortho_loss = torch.mean(ortho_loss)
    del d
    del vol
    del d_flat
    return s, CT_loss, ortho_loss,adj 