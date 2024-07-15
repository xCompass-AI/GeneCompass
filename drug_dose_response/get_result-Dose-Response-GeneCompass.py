import EmbeddingEvaluator

import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda
import os

# For Drug dose response task, we refer CPA model to complete our jobs, 
# the paper is Learning interpretable cellular responses to complex perturbations in high-throughput screens. 
# The code is at:(https://github.com/facebookresearch/CPA)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Path = '/disk1/xCompass/'


# 2. Dose Response(Based on Compositional Perturbation Autoencoder model)
# i.e Before you test this task, please make sure you have got the embedding file(.npy), if not, please generate it(
# case1: if you want to use model provided by us, run get_emb.py to get it,
# case2: if you use your own model, you can generate embedding file by yourself, and place it into appropriate path)



#adata_path = Path + 'drug_dose_response/data/cpa/datasets/GSM_new.h5ad'
adata_path = Path + 'data/biological_contexts/data/cpa/datasets/GSM_new.h5ad'

#pretrained_cpa_model = Path + 'drug_dose_response/model/cpa/cpa_gem_new.pt'
pretrained_cpa_model = Path + 'data/biological_contexts/model/cpa/cpa_gem_new.pt'


save_path = Path + 'drug_dose_response/cpa/result'

## 2.3 GeneCompass

llm_name = 'GeneCompass-0708'

#emb_file = Path + 'tasks/GeneCompass-emb.npy'
emb_file = Path + 'drug_dose_response/GeneCompass-emb-0708.npy'

EmbeddingEvaluator.CPA_result(adata_path, pretrained_cpa_model, llm_name, save_path, emb_file = emb_file)
