#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator
from random import shuffle

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)      #used for modeling in TransE, RotatE, pRotatE
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float) #used for negative_adversarial_sampling only
    parser.add_argument( '--ote_size', default=1, type=int)      #used for OTE 
    parser.add_argument( '--ote_scale', default=0, type=int, choices=[0,1,2])      #used for OTE 
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--regularization_convke', default=0.0000, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    parser.add_argument('--conv_dims', default="20_40_40", type=str) #used for conve and cconve

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-np_ratio', '--neg_pos_ratio', default=1.0, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('--init_embedding', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--schedule_steps', default=10000, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    

    parser.add_argument('--seed', default=-1, type=int)
    
    #gpred
    parser.add_argument('--add_dummy', action='store_true')
    parser.add_argument('--test_split_num', default=1, type=int)
    parser.add_argument('--use_gpred_cxt', action='store_true')
    #parser.add_argument('--gpred_save_mem', action='store_true')
    parser.add_argument('--gpred_merge_scale', default=1000.0, type=float)
    #parser.add_argument('--use_gpred_rev_score', action='store_true')
    parser.add_argument('--gpred_max_ent_num', default=20000, type=int)
    parser.add_argument('--gpred_max_smp_num', default=500, type=int)
    #graph model
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--link_dropout', default=0.0, type=float)
    parser.add_argument('--context_cluster_num', default=4, type=int)
    parser.add_argument('--context_cluster_scale', default=1, type=int)

    #two parameters used in BiGNNPred and RotatED
    parser.add_argument('--update_scoreweight', action='store_true')
    parser.add_argument('--score_weight_scale', default=1.0, type=float)

    #rgnn
    parser.add_argument('--gnn_mode', default='RotatE', type=str, choices=['RotatE', 'TransE', 'Concat', 'None'])
    parser.add_argument('--gnn_type', default='GAT', type=str, choices=['GAT', 'GCN'])
    #parser.add_argument('--use_gnn', action='store_true')
    parser.add_argument('--extra_fc2', action='store_true')
    parser.add_argument('--gnn_layers', default=2, type=int)
    parser.add_argument('--gnn_conv_dims', default="0_3", type=str)
    #parser.add_argument('--init_factor', default=0, type=float, help="special initialization. 0 means default is used")
    parser.add_argument('--init_mode', default='uniform', type=str, choices=['uniform', 'normal', 'xavier_uniform','xavier_normal'])
    parser.add_argument('--negative_sample_head_size', default=16, type=int)
    parser.add_argument('--negative_sample_tail_size', default=32, type=int)
    parser.add_argument('--negative_sample_half_correct', action='store_true')
    parser.add_argument('--score_sigmoid', action='store_true')
    parser.add_argument('--use_bceloss', action='store_true')
    parser.add_argument('--use_softmarginloss', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help="used for bceloss ")
    parser.add_argument('--same_head_tail', action='store_true')
    parser.add_argument('--layerwise_delta_relation', action='store_true')
    parser.add_argument('--use_combine_scores', action='store_true')
    parser.add_argument('--no_residual', action='store_true')

    #parser.add_argument('--gat_heads', default=4, type=int)
    #parser.add_argument('--gatmlp_layers', default=1, type=int)
    args =  parser.parse_args(args)
    args.conv_dims = [int(x) for x in args.conv_dims.split("_") ] 
    args.gnn_conv_dims = [int(x) for x in args.gnn_conv_dims.split("_") ] 
    #assert not(args.tail_only and args.head_only) #only one can be true or both are false
    return args

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    #args.test_batch_size = argparse_dict['test_batch_size']
    #args.use_gnn = argparse_dict['use_gnn']
    #TODO
    #if args.use_gnn:
    #    args.gnn_mode = argparse_dict['gnn_mode']
    #    args.gnn_type = argparse_dict['gnn_type']
    #    args.extra_fc2 = argparse_dict['extra_fc2']
    #    args.gnn_layers = argparse_dict['gnn_layers']
    #    args.conv_dims = argparse_dict['conv_dims']
    #    args.gnn_conv_dims = argparse_dict['gnn_conv_dims']

def save_model(model, optimizer, save_variable_list, args, is_best_model=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    save_path = "%s/best/"%args.save_path if is_best_model else  args.save_path   
    argparse_dict = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(save_path, 'checkpoint')
    )
    if  model.entity_embedding is not None: 
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'entity_embedding'), 
            entity_embedding
        )
        
        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(save_path, 'relation_embedding'), 
            relation_embedding
        )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO if not args.debug else logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def is_better_metric(best_metrics, cur_metrics):
    if best_metrics is None:
        return True
    if best_metrics[-1]['MRR'] < cur_metrics[-1]['MRR']:
        return True
    return False

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        if 'name' in metric:
            logging.info("results from %s"%metric['name'])
        for m in [x for x in metric if x!="name"]:
            logging.info('%s %s at step %d: %f' % (mode, m, step, metric[m]))
    
        
        
def main(args):
    if args.seed != -1:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        if args.do_train and args.do_valid:
            if not os.path.exists("%s/best/"%args.save_path):
                os.makedirs("%s/best/"%args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    train_triples_tsr = torch.LongTensor(train_triples).transpose(0,1) #idx X batch
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples
    #if args.use_gnn:
    #    assert False
    #    #kge_model = GNN_KGEModel(
    #    #    model_name=args.model,
    #    #    nentity=nentity,
    #    #    nrelation=nrelation,
    #    #    hidden_dim=args.hidden_dim,
    #    #    gamma=args.gamma,
    #    #    num_layers=args.gnn_layers,
    #    #    args = args,
    #    #    dropout=args.dropout,
    #    #    double_entity_embedding=args.double_entity_embedding,
    #    #    double_relation_embedding=args.double_relation_embedding,
    #    #)
    #else:
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        args = args,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )
    
    logging.info('Model Configuration:')
    logging.info(str(kge_model))
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
        train_triples_tsr =  train_triples_tsr.cuda()
    #kge_model.build_cxt_triple_map(train_triples)
    if args.do_train:
        # Set training dataloader iterator
        if args.same_head_tail:
            #shuffle train_triples first and no shuffle within dataloaders. So both head and tail will share the same idx
            shuffle(train_triples) 
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset.collate_fn
            )
            
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset.collate_fn
            )
        else:
            train_dataloader_head = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset.collate_fn
            )
            
            train_dataloader_tail = DataLoader(
                TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
                batch_size=args.batch_size,
                shuffle=True, 
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TrainDataset.collate_fn
            )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        #else:
        #    train_dataloader_rel = DataLoader(
        #        TrainDataset(train_triples, nentity, nrelation, 
        #            args.negative_sample_head_size*args.negative_sample_tail_size, 
        #            'rel-batch', 
        #            negative_sample_head_size =args.negative_sample_head_size, 
        #            negative_sample_tail_size =args.negative_sample_tail_size,
        #            half_correct=args.negative_sample_half_correct), 
        #        batch_size=args.batch_size,
        #        shuffle=True, 
        #        num_workers=max(1, args.cpu_num//2),
        #        collate_fn=TrainDataset.collate_fn
        #    )
        #    train_iterator = BidirectionalOneShotIterator.one_shot_iterator(train_dataloader_rel)
        #    tail_only = True
            
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5, last_epoch=-1)
        #if args.warm_up_steps:
        #    warm_up_steps = args.warm_up_steps
        #else:
        #    warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        if 'score_weight' in kge_model.state_dict() and 'score_weight' not in checkpoint['model_state_dict']:
            checkpoint['model_state_dict']['score_weights'] = kge_model.state_dict()['score_weights']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            #warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            current_learning_rate = 0
    elif args.init_embedding:
        logging.info('Loading pretrained embedding %s ...' % args.init_embedding)
        if kge_model.entity_embedding is not None: 
            entity_embedding = np.load(os.path.join(args.init_embedding, 'entity_embedding.npy'))
            relation_embedding = np.load(os.path.join(args.init_embedding, 'relation_embedding.npy'))
            entity_embedding = torch.from_numpy(entity_embedding).to(kge_model.entity_embedding.device)
            relation_embedding = torch.from_numpy(relation_embedding).to(kge_model.relation_embedding.device)
            kge_model.entity_embedding.data[:entity_embedding.size(0)] = entity_embedding
            kge_model.relation_embedding.data[:relation_embedding.size(0)] = relation_embedding
        init_step = 1
        current_learning_rate = 0
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 1
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('learning_rate = %.5f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training

    #loss_func = nn.BCEWithLogitsLoss(reduction="none") if args.use_bceloss else nn.LogSigmoid()
    if args.use_bceloss:
        loss_func = nn.BCELoss(reduction="none")
    elif args.use_softmarginloss:
        loss_func = nn.SoftMarginLoss(reduction="none")
    else:
        loss_func = nn.LogSigmoid()
    #kge_model.cluster_relation_entity_embedding(args.context_cluster_num, args.context_cluster_scale) 
    if args.do_train:
        training_logs = []
        best_metrics = None 
        #Training Loop
        optimizer.zero_grad()
        for step in range(init_step, args.max_steps+1):
            if step % args.update_freq == 1 or args.update_freq == 1:    
                optimizer.zero_grad()
            log = kge_model.train_step(kge_model, train_iterator, train_triples_tsr, loss_func, args)
            if step % args.update_freq == 0:
                optimizer.step()
            
            training_logs.append(log)
            
            #if step >= warm_up_steps:
            #    current_learning_rate = current_learning_rate / 10
            #    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
            #    optimizer = torch.optim.Adam(
            #        filter(lambda p: p.requires_grad, kge_model.parameters()), 
            #        lr=current_learning_rate
            #    )
            #    warm_up_steps = warm_up_steps * 3
            if step % args.schedule_steps == 0:
                scheduler.step()
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    #'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, [metrics])
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, train_triples_tsr, args )
                log_metrics('Valid', step, metrics)
                if is_better_metric(best_metrics, metrics):
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        #'warm_up_steps': warm_up_steps
                    }
                    save_model(kge_model, optimizer, save_variable_list, args, True)
                    best_metrics = metrics
                #kge_model.cluster_relation_entity_embedding(args.context_cluster_num, args.context_cluster_scale) 
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            #'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
    if args.do_valid and args.do_train:
        #load the best model
        best_checkpoint = torch.load("%s/best/checkpoint"%args.save_path)
        kge_model.load_state_dict(best_checkpoint['model_state_dict'])
        logging.info("Loading best model from step %d"%best_checkpoint['step'])
        step = best_checkpoint['step']

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, train_triples_tsr, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, train_triples_tsr, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, train_triples_tsr, args)
        log_metrics('Test', step, metrics)

        
if __name__ == '__main__':
    main(parse_args())
