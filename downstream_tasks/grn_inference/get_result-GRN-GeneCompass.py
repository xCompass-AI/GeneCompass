import EmbeddingEvaluator

import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda
import os

# For GRN task, we refer DeepSEM framework to complete our jobs, 
# the paper is Modeling gene regulatory networks using neural network architectures. 
# The code is at:(https://github.com/HantaoShu/DeepSEM)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#Path = '/disk1/xCompass/'
#Path = '../../../xCompass/'
Path = os.path.abspath(__file__)[:-60]

data_file = Path + 'downstream_tasks/grn_inference/DeepSEM-master/BEELINE-data/inputs/scRNA-Seq/hESC/BL-hESC-ChIP-seq-no-peca-ExpressionData.csv'
net_file = Path + 'downstream_tasks/grn_inference/DeepSEM-master/BEELINE-data/inputs/scRNA-Seq/hESC/BL-hESC-ChIP-seq-no-peca-network.csv'
get_emb = True
n_epochs = 40

# 1. For GRN(Based on DeepSEM-master)
## 1.3 GeneCompass


save_name = Path + 'downstream_tasks/grn_inference/DeepSEM-master/result/GeneCompass_cls_new-emb-0708'
model = 'GeneCompass_cls_new'

#dataset_path = Path + 'grn_inference/data/geneformer/data-Immune-Human/final-datapatch1'
dataset_path = Path + 'data/biological_contexts/data/geneformer/data-Immune-Human/final-datapatch1'

#checkpoint_path = Path + 'grn_inference/model/GeneCompass/55M/id_value_fourPrior/models'
checkpoint_path = Path + 'data/biological_contexts/model/GeneCompass/55M/id_value_fourPrior/models'

emb_file_path = Path + 'downstream_tasks/grn_inference/GeneCompass-emb-0708.npy'

#prior_embedding_path = Path + 'prior_knowledge/'

EmbeddingEvaluator.DeepSEM_result(Path, data_file, net_file, save_name, model, dataset_path, checkpoint_path, n_epochs, get_emb, emb_file_path)
