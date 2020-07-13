import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import logging

class OTE(nn.Module):
    def __init__(self, num_elem, scale_type=0,  **kwargs):
        super(OTE, self).__init__()
        self.num_elem = num_elem
        self.scale_type = scale_type # 1 abs 2 exp

    def forward(self, inputs, inputs_rel, eps=1e-6):
        #inputs: * X num_dim, where num_dim % num_elem == 0
        inputs_size = inputs.size()
        assert inputs_size[:-1] == inputs_rel.size()[:-1]
        num_dim = inputs.size(-1)

        inputs = inputs.view(-1, 1, self.num_elem )
        if self.use_scale :
            rel = inputs_rel.view(-1, self.num_elem ,  self.num_elem + 1)
            outputs = torch.bmm(inputs, rel[:,:,:self.num_elem]*self.get_scale(rel[:,:,self.num_elem:]))
        else:
            rel = inputs_rel.view(-1, self.num_elem ,  self.num_elem )
            outputs = torch.bmm(inputs, rel)
        return outputs.view(inputs_size)
    def forward_rel_index(self, inputs, inputs_rel, rel_idx, eps=1e-6):
        #inputs: * X num_dim, where num_dim % num_elem == 0
        inputs_size = inputs.size()
        #assert inputs_size[:-1] == inputs_rel.size()[:-1]
        num_dim = inputs.size(-1)

        inputs = inputs.view(-1, 1, self.num_elem )
        if self.use_scale :
            rel_size = inputs_rel.size()
            rel = inputs_rel.view(-1, self.num_elem ,  self.num_elem + 1)
            rel = rel[:,:,:self.num_elem]*self.get_scale(rel[:,:,self.num_elem:])
            rel = rel.view(rel_size[0],-1, self.num_elem, self.num_elem).index_select(0, rel_idx)
            rel = rel.view(-1, self.num_elem, self.num_elem)
            outputs = torch.bmm(inputs, rel)
        else:
            rel = inputs_rel.index_select(0, rel_idx)
            rel = rel.view(-1, self.num_elem ,  self.num_elem )
            outputs = torch.bmm(inputs, rel)
        return outputs.view(inputs_size)
        
    
    
    @property
    def use_scale(self):
        return self.scale_type > 0

    def score(self, inputs):
        inputs_size = inputs.size()
        num_dim = inputs.size(-1)
        inputs = inputs.view(-1, num_dim).view(-1, num_dim // self.num_elem, self.num_elem )
        inputs = inputs.view(-1, self.num_elem )
        scores = inputs.norm(dim=-1).view(-1, num_dim // self.num_elem).sum(dim=-1).view(inputs_size[:-1]) 
        return scores 

    def get_scale(self, scale):
        if self.scale_type == 1:
            return scale.abs()
        if self.scale_type == 2:
            return scale.exp()
        raise ValueError("Scale Type %d is not supported!"%self.scale_type)

    def reverse_scale(self, scale, eps=1e-9):
        if self.scale_type == 1:
            return 1/(abs(scale) + eps)
        if self.scale_type == 2:
            return -scale
        raise ValueError("Scale Type %d is not supported!"%self.scale_type)

    def scale_init(self):
        if self.scale_type == 1:
            return 1.0
        if self.scale_type == 2:
            return 0.0
        raise ValueError("Scale Type %d is not supported!"%self.scale_type)

    def orth_embedding(self, embeddings, eps=1e-18, do_test=True):
        #orthogonormalizing embeddings
        #embeddings: num_emb X num_elem X (num_elem + (1 or 0))
        num_emb = embeddings.size(0)
        assert embeddings.size(1) == self.num_elem
        assert embeddings.size(2) == (self.num_elem + (1 if self.use_scale else 0))
        if self.use_scale :
            emb_scale = embeddings[:,:,-1]
            embeddings = embeddings[:,:,:self.num_elem]

        u = [embeddings[:,0]]
        uu = [0]*self.num_elem
        uu[0] = (u[0]*u[0]).sum(dim=-1)
        if do_test and (uu[0] < eps).sum() > 1:
            return None
        u_d = embeddings[:,1:]
        for i in range(1, self.num_elem):
            u_d = u_d - u[-1].unsqueeze(dim=1)*((embeddings[:,i:]*u[i-1].unsqueeze(dim=1)).sum(dim=-1)/uu[i-1].unsqueeze(dim=1)).unsqueeze(-1)
            u_i = u_d[:,0] 
            u_d = u_d[:,1:]
            uu[i] = (u_i*u_i).sum(dim=-1)
            if do_test and (uu[i] < eps).sum() > 1:
                return None
            u.append(u_i)

        u = torch.stack(u,dim=1)    #num_emb X num_elem X num_elem
        u_norm = u.norm(dim=-1,keepdim=True)
        u = u/u_norm
        if self.use_scale :
            u = torch.cat((u, emb_scale.unsqueeze(-1)), dim=-1)
        return u
    def orth_reverse_mat(self, rel_embeddings):
        rel_size = rel_embeddings.size()
        if self.use_scale:
            rel_emb = rel_embeddings.view(-1, self.num_elem,self.num_elem + 1)
            rel_mat = rel_emb[:,:,:self.num_elem].contiguous().transpose(1,2) 
            rel_scale = self.reverse_scale(rel_emb[:,:,self.num_elem:]) 
            rel_embeddings = torch.cat((rel_mat, rel_scale),dim=-1).view(rel_size)
        else:
            rel_embeddings = rel_embeddings.view(-1, self.num_elem,self.num_elem).transpose(1,2).contiguous().view(rel_size)
        return rel_embeddings

    def fix_embedding_rank(self, embeddings, new_rand=False, eps=1e-6, tol=1e-9):
        #make the embeddings full rank
        #embeddings: num_emb X num_elem X num_elem
        num_emb = embeddings.size(0)
        assert embeddings.size(1) == self.num_elem
        assert embeddings.size(2) == self.num_elem + (1 if self.use_scale else 0)
        if self.use_scale:
            emb_scale = embeddings[:,:,-1]
            embeddings = embeddings[:,:,:self.num_elem].contiguous()
            

        for i in range(num_emb):
            mat = embeddings[i]
            if torch.det(mat).abs() < tol:
                logging.warning("Sigular matrix, add initial eps %e"%eps)
                num = 0
                eps_u=eps
                while torch.det(mat).abs() < tol:  
                    if new_rand:
                        mat = torch.rand(mat.size())
                    else:
                        if num >= 1:
                            eps_u = eps_u * 2  
                            #logging.warning("Sigular matrix fixing, increase eps to %e "%eps_u)
                        mat = embeddings[i] +  torch.diag(torch.ones(self.num_elem)*eps_u).to(mat.device)
                        num = num + 1
                embeddings[i].data.copy_(mat)

        if self.use_scale :
            embeddings = torch.cat((embeddings, emb_scale.unsqueeze(-1)), dim=-1)
        return embeddings

            




