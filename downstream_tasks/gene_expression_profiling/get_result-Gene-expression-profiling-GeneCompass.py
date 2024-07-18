import EmbeddingEvaluator

import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda
import os

# For Gene-expression-profiling task, we refer DeepCE model to complete our jobs, 
# the paper is A deep learning framework for high-throughput mechanism-driven phenotype compound screening and its application to COVID-19 drug repurposing. 
# The code is at:(https://github.com/pth1993/DeepCE)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Path = '/disk1/xCompass/'
Path = os.path.abspath(__file__)[:-94]


# 3. Gene expression profiling(Based on DeepCE)
# i.e Before you test this task, please make sure you have got the embedding file(.npy), if not, please generate it(
# case1: if you want to use model provided by us, run get_emb.py to get it,
# case2: if you use your own model, you can generate embedding file by yourself, and place it into appropriate path)

drug_file = './DeepCE-master/DeepCE/data/drugs_smiles.csv'
gene_file = './DeepCE-master/DeepCE/data/gene_vector.csv'
train_file = './DeepCE-master/DeepCE/data/signature_train.csv'
dev_file = './DeepCE-master/DeepCE/data/signature_dev.csv'
test_file = './DeepCE-master/DeepCE/data/signature_test.csv'
dropout = 0.1
batch_size = 32
max_epoch = 120
output_path = './DeepCE-master/DeepCE/out-GeneCompass-emb-0716/output.txt'

## 3.3 GeneCompass

emb_file = Path + 'downstream_tasks/gene_expression_profiling/GeneCompass-emb-0708.npy'

try:
    os.mkdir(output_path[:-11])
except:
    print('dir exist')

EmbeddingEvaluator.DeepCE_result(drug_file, gene_file, train_file, dev_file, test_file, dropout, batch_size, max_epoch, emb_file, output_path)