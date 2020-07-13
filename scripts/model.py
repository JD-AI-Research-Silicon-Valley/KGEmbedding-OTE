#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bignn import BiGNN 
from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader
from rotate import RotatE_Trans
from ote import OTE 
from dataloader import TestDataset
pi = 3.14159265358979323846

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, args,
                 double_entity_embedding=False, double_relation_embedding=False, 
                 dropout=0, init_embedding=True):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0                  #for embedding initialization
        self.scale_relation = True
        
        self.gamma = nn.Parameter(          #for embedding initialization and modeling (RototE, TransE)
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        #used for GPred
        self.add_dummy = args.add_dummy     
        add_dummy = 1 if args.add_dummy else 0
        self.test_split_num = args.test_split_num 
        
        ##(15 + 2) / 200 ~ 0.08 
        ##(9 + 2 ) / 1000 ~ 0.01
        self.embedding_range = nn.Parameter(
            torch.Tensor([0.01]), 
            requires_grad=False
        )
        self._aux = {"edge_index":None}
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        self.ote = None
        if model_name in ( 'OTE', 'BiGNNPredOTE'):
            assert self.entity_dim % args.ote_size == 0
            sub_emb_num = self.entity_dim // args.ote_size  
            self.ote = OTE(args.ote_size, args.ote_scale)
            use_scale = self.ote.use_scale
            self.relation_dim =  self.entity_dim * (args.ote_size + (1 if use_scale else 0) )


            self.relation_embedding = nn.Parameter(torch.zeros(nrelation+add_dummy, self.relation_dim)) 
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=0, 
                b=1.0
            )
            if use_scale:
                self.relation_embedding.data.view(-1, args.ote_size + 1)[:,-1] = self.ote.scale_init()      #start with no scale
            #make initial relation embedding orthogonal
            rel_emb_data = self.orth_rel_embedding()
            self.relation_embedding.data.copy_(rel_emb_data.view(nrelation+add_dummy, self.relation_dim))
        else:
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation+add_dummy, self.relation_dim)) 
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
        if init_embedding:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity+add_dummy, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
            
        else:
            self.entity_embedding = None
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        self.gpred = None
        self.bignn = None
        if model_name in ("BiGNNPred",):
            self.bignn = BiGNN(nentity+add_dummy,nrelation+add_dummy, args, self.embedding_range.item())
            assert self.add_dummy

        if model_name in ("BiGNNPredOTE",):
            self.bignn = BiGNN(nentity+add_dummy,nrelation+add_dummy, args, self.embedding_range.item(), model='OTE')
            assert self.add_dummy
        self.score_sigmoid = args.score_sigmoid
        
        if model_name in ('RotatE', 'RotatED') and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        self.dropout = nn.Dropout(dropout) if dropout >  0 else lambda x: x

    def orth_rel_embedding(self):
            rel_emb_size = self.relation_embedding.size()
            ote_size = self.ote.num_elem
            scale_dim = 1 if self.ote.use_scale else 0
            rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim )
            rel_embedding = self.ote.orth_embedding(rel_embedding).view(rel_emb_size) 
            if rel_embedding is None:
                rel_embedding = self.ote.fix_embedding_rank(self.relation_embedding.view(-1, ote_size, ote_size + scale_dim )) 
                if self.training:
                    self.relation_embedding.data.copy_(rel_embedding.view(rel_emb_size))
                    rel_embedding = self.relation_embedding.view(-1, ote_size, ote_size + scale_dim )
                rel_embedding = self.ote.orth_embedding(rel_embedding, do_test=False).view(rel_emb_size) 
            return rel_embedding 

        
    def cal_embedding(self, edge_index):
        self._aux['edge_index'] = edge_index
        if self.model_name in ( "OTE", "BiGNNPredOTE"):
            rel_embedding = self.orth_rel_embedding()
            self._aux['rel_emb'] = rel_embedding
            self._aux['ent_emb'] = self.entity_embedding
    def get_embedding(self):
        if self.model_name in ( "OTE", "BiGNNPredOTE"):
            return self._aux['rel_emb'], self._aux['ent_emb']
        return self.relation_embedding, self.entity_embedding

    def reset_embedding(self):
        for k in [ key for key in self._aux.keys() if key != "static"]:
            self._aux[k] = None
        pass
    def get_model_func(self):
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotatED': self.RotatED,
            'pRotatE': self.pRotatE,
            'BiGNNPred': self.BiGNNPred,
            'BiGNNPredOTE': self.BiGNNPredOTE,
            'OTE':    self.OTE,
        }
        return model_func

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        relation_embedding, entity_embedding = self.get_embedding()
        if mode in ('single', 'head-single', 'tail-single'):
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            head_ids = sample[:,0].unsqueeze(1) if mode == 'head-single' else sample[:,0]
            tail_ids = sample[:,2].unsqueeze(1) if mode == 'tail-single' else sample[:,2]
            self._aux['samples'] = (head_ids, sample[:,1], tail_ids, mode)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            self._aux['samples'] = (head_part, tail_part[:,1], tail_part[:,2], mode)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            self._aux['samples'] = (head_part[:,0], head_part[:,1], tail_part, mode)
        
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = self.get_model_func() 
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score


    def RotatE(self, head, relation, tail, mode):
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi) if self.scale_relation else relation 

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score, im_score = RotatE_Trans((re_tail, im_tail), (re_relation, im_relation), False)
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score, im_score = RotatE_Trans((re_head, im_head), (re_relation, im_relation), True)
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def OTE(self, head, relation, tail, mode):
        if mode in ("head-batch",  'head-single'):
            relation = self.ote.orth_reverse_mat(relation)
            output = self.ote(tail, relation)
            score = self.ote.score(output-head)
        else:
            output = self.ote(head, relation)
            score = self.ote.score(output-tail)
        score = self.gamma.item() - score

        return score
   
    #return hr or tr
    def RotatED(self, head, relation, tail, mode):
        
        neg_sz  = max(head.size(1), tail.size(1))
        bsz = head.size(0)
        rel_idx = self._aux['samples'][1]
        swi = 0 if mode in ("head-batch",  'head-single') else 1

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi) if self.scale_relation else relation 

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        re_score, im_score = RotatE_Trans((re_tail, im_tail), (re_relation, im_relation), False)
        re_score = re_score - re_head
        im_score = im_score - im_head
        score = torch.stack([re_score, im_score], dim = 0).norm(dim=0).sum(dim=2) #*score_weight[0]

        re_score, im_score = RotatE_Trans((re_head, im_head), (re_relation, im_relation), True)
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = score +  torch.stack([re_score, im_score], dim = 0).norm(dim=0).sum(dim=2) #*score_weight[1]
        score = self.gamma.item() - score
        return score

    def pRotatE(self, head, relation, tail, mode):
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def BiGNNPred(self, head, relation, tail, mode):    
        pi = 3.14159265358979323846

        neg_sz  = max(head.size(1), tail.size(1))
        bsz = head.size(0)
        rel_idx = self._aux['samples'][1]
        swi = 0 if mode in ("head-batch",  'head-single') else 1

        if self.training or not ( 'bignn_embedding' in self._aux and self._aux['bignn_embedding'] is not None ) : 
            hr_c_all, tr_c_all = self.bignn(self.entity_embedding, self.relation_embedding,  
                        self._aux['edge_index'])
            if not self.training:
                self._aux['bignn_embedding'] = (hr_c_all, tr_c_all )
        else:
            hr_c_all, tr_c_all = self._aux['bignn_embedding']

        rel_ids =  self._aux['samples'][1] if self._aux['samples'][2].dim() == 1 else self._aux['samples'][1].unsqueeze(1).repeat(1, neg_sz)
        hr_c = self.bignn.retrival_emb(hr_c_all, self._aux['samples'][2], rel_ids, True)
        rel_ids =  self._aux['samples'][1] if self._aux['samples'][0].dim() == 1 else self._aux['samples'][1].unsqueeze(1).repeat(1, neg_sz)
        tr_c = self.bignn.retrival_emb(tr_c_all, self._aux['samples'][0], rel_ids, False)
        
        #tr - h 
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi) if self.scale_relation else relation 

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score, im_score = RotatE_Trans((re_tail, im_tail), (re_relation, im_relation), False)
        tr = torch.cat((re_score, im_score), dim=-1)

        re_score = re_score - re_head
        im_score = im_score - im_head
        score = self.dropout(torch.stack([re_score, im_score], dim = 0).norm(dim=0)).sum(dim=2)
        
        #add tr - tr_c
        re_d, im_d = torch.chunk((tr - tr_c), 2, dim=-1)
        score = score + self.dropout(torch.stack([re_d, im_d], dim=0).norm(dim=0)).sum(dim=2)
        
        #hr - t
        re_score, im_score = RotatE_Trans((re_head, im_head), (re_relation, im_relation), True)
        hr = torch.cat((re_score, im_score), dim=-1)

        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = score +  self.dropout(torch.stack([re_score, im_score], dim = 0).norm(dim=0)).sum(dim=2)

        #add hr - hr_c
        re_d, im_d = torch.chunk((hr - hr_c), 2, dim=-1)
        score = score + self.dropout(torch.stack([re_d, im_d], dim=0).norm(dim=0)).sum(dim=2)

        score = self.gamma.item() - score/4.0
        return score

    def BiGNNPredOTE(self, head, relation, tail, mode):    

        neg_sz  = max(head.size(1), tail.size(1))
        bsz = head.size(0)
        rel_idx = self._aux['samples'][1]
        swi = 0 if mode in ("head-batch",  'head-single') else 1
        relation_embedding, entity_embedding = self.get_embedding()

        if self.training or not ( 'bignn_embedding' in self._aux and self._aux['bignn_embedding'] is not None ) : 
            hr_c_all, tr_c_all = self.bignn(entity_embedding, relation_embedding,  
                        self._aux['edge_index'])
            if not self.training:
                self._aux['bignn_embedding'] = (hr_c_all, tr_c_all )
        else:
            hr_c_all, tr_c_all = self._aux['bignn_embedding']

        rel_ids =  self._aux['samples'][1] if self._aux['samples'][2].dim() == 1 else self._aux['samples'][1].unsqueeze(1).repeat(1, neg_sz)
        hr_c = self.bignn.retrival_emb(hr_c_all, self._aux['samples'][2], rel_ids, True)
        rel_ids =  self._aux['samples'][1] if self._aux['samples'][0].dim() == 1 else self._aux['samples'][1].unsqueeze(1).repeat(1, neg_sz)
        tr_c = self.bignn.retrival_emb(tr_c_all, self._aux['samples'][0], rel_ids, False)
        
        hr_rel = relation 
        tr_rel = self.ote.orth_reverse_mat(relation) 

        #tr - h 
        tr = self.ote(tail, tr_rel) if tail.size(1) == 1 else self.ote(tail, tr_rel.expand(-1, neg_sz, -1).contiguous())

        score = self.ote.score(tr-head)
        
        #add tr - tr_c
        score = score + self.ote.score(tr - tr_c) 
        
        #hr - t
        hr = self.ote(head, hr_rel) if head.size(1) == 1 else self.ote(head, hr_rel.expand(-1, neg_sz, -1).contiguous()) 

        score = score +  self.ote.score(hr - tail) 

        #add hr - hr_c
        score = score + self.ote.score(hr - hr_c) 

        score = self.gamma.item() - score/4.0
        return score


    
    @staticmethod
    def apply_loss_func(score, loss_func, is_negative_score=False, label_smoothing=0.1):
        if isinstance(loss_func, nn.SoftMarginLoss):
            tgt = -1 if is_negative_score else 1
            tgt = torch.empty(score.size()).fill_(tgt).to(score.device)
            output = loss_func(score, tgt)
        elif isinstance(loss_func, nn.BCELoss):
            #bceloss
            tgt = 0 if is_negative_score else 1
            if label_smoothing > 0:
                tgt = tgt*(1-label_smoothing)+0.0001
            tgt = torch.empty(score.size()).fill_(tgt).to(score.device)
            output = loss_func(score, tgt)
        else:
            output = loss_func(-score) if is_negative_score else loss_func(score) 
        return output
            
            
    @staticmethod
    def train_step(model,  train_iterator, edge_index, loss_func,  args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()


        positive_sample, negative_sample, subsampling_weight, mode, idxs = next(train_iterator)
        
        #remove target links from graph
        num_edges = edge_index.size(1)
        select_index = torch.ones(num_edges).byte().to(edge_index.device)
        select_index[idxs] = 0 
        edge_index = torch.index_select(edge_index, dim=1, index=select_index.nonzero().view(-1))
        
        if args.link_dropout > 0: 
            smp_link = torch.rand(edge_index.size(1), device=edge_index.device).gt(args.link_dropout).nonzero().view(-1)
            edge_index = torch.index_select(edge_index, 1, smp_link,)
        model.cal_embedding(edge_index)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            if isinstance(negative_sample, tuple):
                negative_sample = [ x.cuda() for x in negative_sample ]
            else:
                negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
        
        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.debug:
            logging.debug("Train (%s): score mean %.4f max %.4f"%(mode, negative_score.mean().item(), negative_score.max().item()))

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing)).sum(dim = 1)
        else:
            negative_score = model.apply_loss_func(negative_score, loss_func, True, args.label_smoothing).mean(dim = 1)
        pmode="head-single" if mode == "head-batch" else "tail-single"
        positive_score = model(positive_sample, pmode)
        if args.debug:
            logging.debug("Train: positive sample score mean %.4f max %.4f"%(positive_score.mean().item(), positive_score.max().item()))

        positive_score = model.apply_loss_func(positive_score, loss_func,False, args.label_smoothing).squeeze(dim = 1)
        loss_sign = 1 if args.use_bceloss or args.use_softmarginloss else -1
        if args.uni_weight:
            positive_sample_loss = loss_sign* positive_score.mean()
            negative_sample_loss = loss_sign* negative_score.mean()
        else:
            positive_sample_loss = loss_sign* (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = loss_sign* (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss*args.neg_pos_ratio)/(1+args.neg_pos_ratio)
        

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()


        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }
        model.reset_embedding()

        return log
    def split_test(self, sample, mode):
        if self.test_split_num == 1:
            return self(sample, mode)
        p_sample, n_sample = sample
        scores = []
        sub_samples = torch.chunk(n_sample, self.test_split_num, dim=1)
        for n_ss in sub_samples:
            scores.append(self((p_sample, n_ss.contiguous()), mode))
        return torch.cat(scores, dim=1)

    @staticmethod
    def test_step(model, test_triples, all_true_triples, edge_index, args, head_only=False, tail_only=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        assert ('bignn_embedding' not in model._aux) or (model._aux['bignn_embedding']  is None) 
        model.cal_embedding(edge_index)
        #model.sample_clusters_indicator(None, num_edges)
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            if head_only:
                test_dataset_list = [test_dataloader_head ]
            elif tail_only:
                test_dataset_list = [test_dataloader_tail]
            else:
                test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = [ [] for i in test_dataset_list ]

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for k, test_dataset in enumerate(test_dataset_list):
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        #model.set_pos_sample(positive_sample,  mode)
                        score = model.split_test((positive_sample, negative_sample), mode)
                        if args.debug:
                            logging.debug("Eval: ALL score mean: %.4f, max: %.4f"%( score.mean().item(), score.max().item()))
                            pindex = positive_sample[:,0] if mode == 'head-batch' else positive_sample[:,2]
                            pscore = torch.gather(score, 1, pindex.unsqueeze(1) )
                            logging.debug("Eval: Positive samples (%s) mean %.4f, max: %.4f"%
                                (mode, pscore.mean().item(), pscore.max().item())) 
                        score += filter_bias*(score.max() - score.min())

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs[k].append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })
                            buf ="%s "%mode + "rank %d "%ranking + " ".join(("%s"%int(x) for x in positive_sample[i])) + "\t" # '[ %d %d %d ]\t'%(int(x) for x in positive_sample[i])
                            buf = buf + " ".join( ["%d"%x for x in argsort[i][:10]])
                            logging.debug(buf)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = [ {} for l in logs ]
            for i, log in enumerate(logs):
                for metric in log[0].keys():
                    metrics[i][metric] = sum([lg[metric] for lg in log])/len(log)
                if len(logs) > 1:
                    metrics[i]['name'] = "head-batch" if i == 0 else "tail-batch"
            if len(logs)==2:
                metrics_all = {}
                log_all = logs[0] + logs[1] 
                for metric in log_all[0].keys():
                    metrics_all[metric] = sum([lg[metric] for lg in log_all])/len(log_all)
                metrics_all['name'] = "Overall"
                metrics.append(metrics_all)

        model.reset_embedding()
        return metrics

