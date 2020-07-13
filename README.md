# KGEmbedding-OTE
Knowledge Graph Embedding - Orthogonal Relation Transforms with Graph Context Modeling for Knowledge Graph Embedding. accepted by ACL 2020

1. Background

Knowledge Graph is a large-scale semantic network that stores human knowledge in the form of graph, where node represents entity, and edge indicates the relationship between two entities. In practice, knowledge graph is usually expressed in the form of a triplet to state the facts. The triplet is generally recorded as (head entity, relation, tail entity), such as (Yao Ming, place of birth, Shanghai). Knowledge graphs have been widely and successfully applied in many fields such as Information Retrieval, Question Answering, Recommendation Systems, and domain-specific applications (e.g., medicine, finance and education). 

Although there are many large-scale, open-domain knowledge graphs in practice, these knowledge graphs still have far from being complete, and also need to be updated with new facts periodically. Therefore, the task of knowledge graph completion is particularly important, and has attracted much attention from the academic community and the industry. In recent years, the embedded representation learning of knowledge graph has become the mainstream method for knowledge graph completion. It maps entities and relations to vectors/matrix/tensors in the low-dimensional continuous space, and scores each triple based on these low-dimensional representations as the probability of the triple being true. The graph embedding representations are divided into two categories according to the different scoring mechanisms: distance-based models and semantic matching-based models.

A good knowledge graph embedding should model: (1) symmetric (e.g., similarity), inverse (e.g., father vs son) and compositional (e.g., grandpaí(father, father)) relations; and (2) 1-to-N, N-to-1, and N-to-N relation predictions, such as, head entity "Ang Lee" connected to multiple tail entities "Chosen, Hulk, Life of Pi" via relation "director_of", it makes the 1-to-N prediction hard because the mapping from certain entity-relation pair could lead to multiple different entities.

2. Orthogonal Relation Transform base KG Embedding

In this work, we propose a novel distance-based approach for knowledge graph link prediction. First, we utilize the group-based orthogonal transform embedding to represent relations which is able to model symmetric, inverse and compositional relations while achieves better modeling capacity. Second, we integrate the graph context into distance scoring functions directly to handle the complex 1-to- N, N-to-1, and N-to-N relation prediction. Specifically, graph context is explicitly modeled via two directed context representations. Each entity embedding in knowledge graph is augmented with two context representations, which are computed from the neighboring outgoing and incoming entities/edges respectively. The proposed approach improves prediction accuracy on the difficult 1-to-N, N-to-1 and N-to-N cases. Our experimental results show that it achieves state-of-the-art results on two common benchmarks FB15k-237 (from Google's Freebase) and WNRR-18 (from WordNet), especially on FB15k-237 which has many high in-degree nodes.

3. Source codes 

#  Graph Based linking prediction 
**Introduction**

The source code for "Context Enhanced Orthogonal Linear Embeddings for Knowledge GraphLink Prediction" accepted by ACL 2020.

**Implemented features**

(1) Models:
 - [x] RotatE
 - [x] TransE
 - [x] OTE 
 - [x] GC-OTE (BiGNNPredOTE) 

(2) Evaluation Metrics:

 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 (filtered)

(3) Loss Function:

 - [x] Uniform Negative Sampling
 - [x] Self-Adversarial Negative Sampling

**Usage**

(4) Knowledge Graph Data:
 - *entities.dict*: a dictionary map entities to unique ids
 - *relations.dict*: a dictionary map relations to unique ids
 - *train.txt*: the KGE model is trained to fit this data set
 - *valid.txt*: create a blank file if no validation data is available
 - *test.txt*: the KGE model is evaluated on this data set

**Recipes**

(5) Settings
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

