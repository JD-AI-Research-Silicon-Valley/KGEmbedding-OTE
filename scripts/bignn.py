import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing 
from torch_geometric.utils import softmax as scatter_softmax 
from torch_geometric.utils import scatter_
from rotate import RotatE_Trans
from ote import OTE 
import math

#GNN with direction
class DGNNLayer(nn.Module):
    def __init__(self, args, is_head_rel=True, **kwargs):
        super(DGNNLayer, self).__init__()
        if args.gnn_type == "GAT":
            self.gnn_model = self.GAT_update 
        elif args.gnn_type == "GCN":
            self.gnn_model = self.GCN_update 
        else:
            raise NotImplementedError("Not implementation")

        self.is_head_rel = is_head_rel #is_head_rel==True:  update tail from head + relation
                                       #is_head_rel==False: update head from tail + relation

    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        ent_ins = torch.index_select(entities, 0, edge_index[pos])
        return ent_ins 

    def GCN_update(self, ent_ins, edge_index, entities ):
        ent_pos = 2 if self.is_head_rel else 0  #ent_pos: position of ent to be upated
        x_ents = scatter_("mean", ent_ins, edge_index[ent_pos], entities.size(0)) 
        return x_ents 

    def GAT_update(self, ent_ins, edge_index, entities):
        def _GAT_update(k, qv, idx,  entities):
            embed_dim = k.size(-1)
            alpha = (k*qv).sum(dim=-1)/math.sqrt(embed_dim)
            attn_wts = scatter_softmax(alpha.unsqueeze(1), idx, entities.size(0))
            wqv = qv*attn_wts 
            out = scatter_("add", wqv, idx, dim_size = entities.size(0)) 
            return out
        ent_pos = 2 if self.is_head_rel else 0  #ent_pos: position of ent to be upated
        ent_ori = torch.index_select(entities, 0,  edge_index[ent_pos])
        x_ents = _GAT_update(ent_ori, ent_ins, edge_index[ent_pos], entities)
        return x_ents
        
    def forward(self, entities, relations, edge_index):
        ent_ins = self.Calculate_Entity(entities, relations, edge_index)
        x_ents = self.gnn_model(ent_ins,  edge_index, entities) 
        return x_ents

class RotatEDGNNLayer(DGNNLayer):

    def __init__(self, args, dummy_node_id, embedding_range=1.0, is_head_rel=True, **kwargs):
        super(RotatEDGNNLayer, self).__init__(args, is_head_rel)
        self.embedding_range = embedding_range
        self.dummy_node_id = dummy_node_id

    def do_trans(self, ent, rel ): 
        pi = 3.14159262358979323846
        phase_relation = rel/(self.embedding_range/pi)  
        re_rel = torch.cos(phase_relation)  
        im_rel = torch.sin(phase_relation)
        re_ent, im_ent = torch.chunk(ent, 2, dim=-1)
         
        if self.is_head_rel: #ent is head, tail-batch
            re_score = re_ent * re_rel - im_ent * im_rel
            im_score = re_ent * im_rel + im_ent * re_rel
        else:   #ent is tail, head-batch
            re_score = re_rel * re_ent + im_rel * im_ent
            im_score = re_rel * im_ent - im_rel * re_ent
        scores = torch.cat((re_score, im_score), dim=-1)
        return scores
    def do_trans_rel_index(self, ent, rel, rel_idx):
        rel = rel.index_select(0, rel_idx)
        return self.do_trans(ent, rel)

    def eff_do_trans(self, ent_ids, rel_ids, entities, relation):
        def _get_abs_idx(idx1, idx2, stride):
            return idx1*stride + idx2
        def _extr_abs_idx(abs_idx, stride):
            idx1 = abs_idx / stride
            idx2 = abs_idx.fmod(stride)
            return idx1, idx2
        num_entity = entities.size(0)
        num_relation = relation.size(0)
        abs_idx = _get_abs_idx(ent_ids, rel_ids, num_relation) 
        uniq_idx = torch.unique(abs_idx)
        to_uniq_table = torch.zeros( num_entity*num_relation, device=uniq_idx.device, dtype=torch.int64) 
        to_uniq_table[uniq_idx] = torch.arange(len(uniq_idx), device=uniq_idx.device)
        uniq_ent_ids, uniq_rel_ids = _extr_abs_idx(uniq_idx, num_relation)
        #ent_out_uniq = self.do_trans(entities.index_select(0,uniq_ent_ids), 
        #            relation.index_select(0,uniq_rel_ids))
        ent_out_uniq = self.do_trans_rel_index(entities.index_select(0,uniq_ent_ids), 
                    relation, uniq_rel_ids)
        ent_out = ent_out_uniq.index_select(0, to_uniq_table[abs_idx])
        return ent_out
        

    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        tnp = 2 if self.is_head_rel else 0
        #rel_ins = torch.index_select(relation, 0, edge_index[1])
        #ent_ins = torch.index_select(entities, 0, edge_index[pos])
        ent_out = self.eff_do_trans(edge_index[pos], edge_index[1], entities, relation)

        return ent_out 

class OTEDGNNLayer(RotatEDGNNLayer):
    def __init__(self, args, dummy_node_id, embedding_range=1.0, is_head_rel=True, **kwargs):
        super(OTEDGNNLayer, self).__init__(args, dummy_node_id, embedding_range, is_head_rel, **kwargs)
        self.ote = OTE(args.ote_size, args.ote_scale)

    def do_trans(self, ent, rel ): 
        output = self.ote(ent, rel.contiguous())
        return output 

    def do_trans_rel_index(self, ent, rel, rel_idx):
        output = self.ote.forward_rel_index(ent, rel, rel_idx)
        return output

    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        tnp = 2 if self.is_head_rel else 0
        #rel_ins = torch.index_select(relation, 0, edge_index[1])
        #ent_ins = torch.index_select(entities, 0, edge_index[pos])
        hr_rel, tr_rel = torch.chunk(relation, 2, dim=1)
        relation = relation if self.is_head_rel else self.ote.orth_reverse_mat(relation) 
        ent_out = self.eff_do_trans(edge_index[pos], edge_index[1], entities, relation)
        return ent_out
    
class BiGNN(nn.Module):
    def __init__(self,  num_entity, num_rels,  args, embedding_range=1.0, model="RotatE" ):
        super(BiGNN, self).__init__()
        self.num_entity = num_entity
        self.num_rels = num_rels
        if model == 'RotatE':
            self.head_rel = RotatEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=True)
            self.tail_rel = RotatEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=False)
        elif model == 'OTE':
            self.head_rel = OTEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=True)
            self.tail_rel = OTEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=False)
        else:
            raise ValueError("Not defined!")

    @property
    def dummy_ent_id(self):
        return self.num_entity - 1
    @property
    def dummy_rel_id(self):
        return self.num_rels - 1

    def retrival_emb(self, ent_emb, ent_id, rel_id, is_hr=True):
        if ent_id.dim() == 1:
            return ent_emb.index_select(0, ent_id).unsqueeze(1)
        bsz, neg_sz = ent_id.size()
        return ent_emb.index_select(0,ent_id.view(-1)).view(bsz, neg_sz, -1)

    def forward(self, entities, relation,  edge_index):
        #calculate hr
        ##add dummy_head
        edge_dummy_idx = [ torch.LongTensor([self.dummy_ent_id, self.dummy_rel_id, i]) for i in range(self.num_entity) ] 
        edge_dummy_idx = torch.stack(edge_dummy_idx).transpose(0,1).to(edge_index.device) #3 X N
        edge_aug_idx = torch.cat((edge_index, edge_dummy_idx), dim=1)  
        hr = self.head_rel(entities, relation, edge_aug_idx)
        #calculate tr
        ##add dummy tail
        edge_dummy_idx = [ torch.LongTensor([i, self.dummy_rel_id, self.dummy_ent_id]) for i in range(self.num_entity) ] 
        edge_dummy_idx = torch.stack(edge_dummy_idx).transpose(0,1).to(edge_index.device) #3 X N
        edge_aug_idx = torch.cat((edge_index, edge_dummy_idx), dim=1)  
        tr = self.tail_rel(entities, relation, edge_aug_idx)
        return hr, tr

            
