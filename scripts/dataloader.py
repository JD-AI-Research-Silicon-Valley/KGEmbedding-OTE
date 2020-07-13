#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import collections

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, negative_sample_head_size=1, negative_sample_tail_size=2, half_correct=False):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.count_rels = self.count_rel_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.rel_tail = self.get_true_relation_to_tail(self.triples)
        #only valid for rel-batch mode
        self.negative_sample_head_size = negative_sample_head_size    
        self.negative_sample_tail_size = negative_sample_tail_size    
        if mode == 'rel-batch':
            assert self.negative_sample_head_size >= 1
            assert self.negative_sample_tail_size >= 1
        self.half_correct = half_correct
        
    def __len__(self):
        return self.len
    def sample_negative_sample(self, sample_size, head, rel, tail,  mode):
        negative_sample_size = 0 
        negative_sample_list = []
        half_correct = self.half_correct
        while negative_sample_size < sample_size:
            negative_sample = np.random.randint(self.nentity, size=sample_size*2)
            if mode == 'head-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_head[(rel, tail)], 
                    assume_unique=True, 
                    invert=True
                )
            elif mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample, 
                    self.true_tail[(head, rel)], 
                    assume_unique=True, 
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:sample_size]
        return negative_sample

        
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        #negative_sample_list = []
        #negative_sample_size = 0
        if self.mode == 'rel-batch':
            subsampling_weight = self.count_rels[relation]
            subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
            #sample head
            negative_sample_head = np.random.randint(1,size=1) 
            while negative_sample_head.size < self.negative_sample_head_size:
                negative_sample_head=np.random.randint(self.nentity, size=self.negative_sample_head_size*5)
                negative_sample_head = negative_sample_head[(negative_sample_head!=head)+1] 
                if negative_sample_head.size > self.negative_sample_head_size:
                    negative_sample_head[0]=head
                    negative_sample_head = negative_sample_head[:self.negative_sample_head_size]
                    break
            negative_sample_tails = []
            for i in range(self.negative_sample_head_size):
                negative_sample_tail = self.sample_negative_sample(
                    self.negative_sample_tail_size,
                    negative_sample_head[i], relation, -1,
                    'tail-batch')
                negative_sample_tails.append(negative_sample_tail)
            negative_sample_tail = np.stack(negative_sample_tails, axis=0) 
            negative_sample = (torch.from_numpy(negative_sample_head), 
                               torch.from_numpy(negative_sample_tail))
        else:
            negative_sample = self.sample_negative_sample(
                self.negative_sample_size,
                head, relation, tail,
                self.mode)

            negative_sample = torch.from_numpy(negative_sample)
        
        positive_sample = torch.LongTensor(positive_sample)
            
        return positive_sample, negative_sample, subsampling_weight, self.mode, idx
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        if isinstance(data[0][1], tuple):
            negative_sample_head = torch.stack([_[1][0] for _ in data], dim=0)
            negative_sample_tail = torch.stack([_[1][1] for _ in data], dim=0)
            negative_sample = (negative_sample_head, negative_sample_tail)
        else:
            negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        idxs = [_[4] for _ in data ]
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight,  mode,  idxs
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def count_rel_frequency(triples):
        relations = [ x[1] for x in triples]
        counter=collections.Counter(relations)
        return counter
        
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    @staticmethod
    def get_true_relation_to_tail(triples):
        '''
        Build a dictionary of true tails given the relation 
        '''
        
        tail_map = {}

        for _, relation, tail in triples:
            if relation not in tail_map:
                tail_map[relation]=[tail]
            else:
                tail_map[relation].append(tail)
        for rel in tail_map.keys():
            tail_map[rel] = np.array(list(set(tail_map[rel])))
        return tail_map 

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
