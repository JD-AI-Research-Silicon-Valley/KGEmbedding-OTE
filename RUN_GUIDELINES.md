#  Graph Based linking prediction 
**Introduction**

The source code for "Context Enhanced Orthogonal Linear Embeddings for Knowledge GraphLink Prediction" accepted by ACL 2020.

**Implemented features**

Models:
 - [x] RotatE
 - [x] TransE
 - [x] OTE 
 - [x] GC-OTE (BiGNNPredOTE) 

Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)

Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Recipes**

OTE model on FB15k-237 dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k-237 \
 --model OTE \
 -n 256 -b 500 -d 400  \
 -g 15.0 -a 0.5 -adv \
 -lr 0.002  --max_steps 250000 \
 --test_batch_size  16  \ 
 --seed 999 --ote_size 20 \
 --log_steps 1000 --valid_steps 10000  \
 --schedule_steps 25000 \
 --ote_scale 2 \
 -save models/OLE_FB15k-237 
```
GC-OTE with OTE seed model for FB15k-237 
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/FB15k-237 \
 --model BiGNNPredOTE \
 -n 256 -b 100 -d 400  \
 -g 12.0 -a 1.4 -adv \
 -lr 0.0002  --max_steps 60000 \
 --test_batch_size  16  \ 
 --seed 999 --ote_size 20 \
 --log_steps 1000 --valid_steps 5000  \
 --schedule_steps 8000 \
 --ote_scale 2 \
 --add_dummy --test_split_num 10 \
 --init_embedding models/OTE_FB15k-237/best \
 -save models/GC_OTE_FB15k-237 
```

OTE model on wn18rr dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/wn18rr  \
 --model OTE \
 -n 256 -b 500 -d 400  \
 -g 5.0 -a 1.8 -adv \
 -lr 0.001  --max_steps 250000 \
 --test_batch_size  16  \ 
 --seed 999 --ote_size 4  \
 --log_steps 1000 --valid_steps 10000  \
 --schedule_steps 25000 \
 --ote_scale 2 \
 -save models/OTE_wn18rr 
```

GC-OTE with OTE seed model for wn18rr with GPU 0 
```
CUDA_VISIBLE_DEVICES=0 python -u scripts/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/wn18rr  \
 --model BiGNNPredOTE \
 -n 256 -b 100 -d 400  \
 -g 10.0 -a 0.5 -adv \
 -lr 0.00003  --max_steps 60000 \
 --test_batch_size  16  \ 
 --seed 999 --ote_size 4  \
 --log_steps 1000 --valid_steps 5000  \
 --schedule_steps 8000 \
 --ote_scale 2 \
 --add_dummy --test_split_num 10 \  
 --init_embedding models/OLE_wn18rr/best \
 -save models/GC_OTE_wn18rr 
```
   Check argparse configuration at scripts/run.py for more arguments and more details.

**Test**
```
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u scripts/run.py --do_test --cuda -init $SAVE

```

**Acknowledge**

The code is based on [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding.git) 
