import EmbeddingGenerator
import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda
import os
import sys
sys.path.append("../")

def set_determenistic_mode(SEED):
    torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)                             # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)             
    torch.cuda.manual_seed_all(SEED) 

set_determenistic_mode(4)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Path = '/disk1/xCompass/'



# GeneCompass_embedding generation


dataset_path = Path + 'data/biological_contexts/data/geneformer/data-Immune-Human/final-datapatch1'
#checkpoint_path = Path + 'tasks/model/GeneCompass/55M/human_4prior_ids_values/models'
checkpoint_path = Path + 'data/biological_contexts/model/GeneCompass/55M/id_value_fourPrior/models'
emb_file_path = Path + 'gene_expression_profiling/GeneCompass-emb-0708.npy'


EmbeddingGenerator.get_GeneCompass_cls_new_embedding(Path = Path, dataset_path=dataset_path, checkpoint_path=checkpoint_path, get_emb = True, emb_file_path = emb_file_path)
