import os
import pickle

import torch
import numpy as np

# PRIOR_EMBEDDING_PATH = '/data/home/ia00/prior_embedding/'
PRIOR_EMBEDDING_PATH = '/data/5_folder/prior_embedding/'

PROMOTER_HOME_PATH = PRIOR_EMBEDDING_PATH + 'promoter_emb/'
NAME_TO_PROMOTER_HUMAN_PATH = PROMOTER_HOME_PATH + 'human_emb_768.pickle'
NAME_TO_PROMOTER_MOUSE_PATH = PROMOTER_HOME_PATH + 'mouse_emb_768.pickle'

COEXP_HOME_PATH = PRIOR_EMBEDDING_PATH + 'gene_co_express_emb/'
NAME_TO_COEXP_HUMAN_PATH = COEXP_HOME_PATH + 'Human_dim_768_gene_28291_random.pickle'
NAME_TO_COEXP_MOUSE_PATH = COEXP_HOME_PATH + 'Mouse_dim_768_gene_27444_random.pickle'

FAMILY_HOME_PATH = PRIOR_EMBEDDING_PATH + 'gene_family/'
NAME_TO_FAMILY_HUMAN_PATH = FAMILY_HOME_PATH + 'Human_dim_768_gene_28291_random.pickle'
NAME_TO_FAMILY_MOUSE_PATH = FAMILY_HOME_PATH + 'Mouse_dim_768_gene_27934_random.pickle'

PECA_HOME_PATH = PRIOR_EMBEDDING_PATH + 'PECA2vec/'
NAME_TO_PECA_HUMAN_PATH = PECA_HOME_PATH + 'human_PECA_vec.pickle'
NAME_TO_PECA_MOUSE_PATH = PECA_HOME_PATH + 'mouse_PECA_vec.pickle'

# TOKEN_DICTIONARY_PATH = PRIOR_EMBEDDING_PATH + 'h&m_token1000W.pickle'
TOKEN_DICTIONARY_PATH = '/data/5_folder/6000W_pretrain_data_merge/human_mouse_tokens.pickle'
ID_TO_NAME_HUMAN_MOUSE_PATH = PRIOR_EMBEDDING_PATH + 'gene_list/Gene_id_name_dict_human_mouse.pickle'

HOMOLOGOUS_GENE_PATH = PRIOR_EMBEDDING_PATH + 'tongyuan_h&m_token.pickle'

def load_prior_embedding(
    name2promoter_human_path=NAME_TO_PROMOTER_HUMAN_PATH, 
    name2promoter_mouse_path=NAME_TO_PROMOTER_MOUSE_PATH, 
    name2coexp_human_path=NAME_TO_COEXP_HUMAN_PATH, 
    name2coexp_mouse_path=NAME_TO_COEXP_MOUSE_PATH,
    name2family_human_path=NAME_TO_FAMILY_HUMAN_PATH, 
    name2family_mouse_path=NAME_TO_FAMILY_MOUSE_PATH,
    name2peca_human_path=NAME_TO_PECA_HUMAN_PATH,
    name2peca_mouse_path=NAME_TO_PECA_MOUSE_PATH,
    id2name_human_mouse_path=ID_TO_NAME_HUMAN_MOUSE_PATH,
    token_dictionary_or_path=TOKEN_DICTIONARY_PATH,
    homologous_gene_path=HOMOLOGOUS_GENE_PATH
):
    if type(token_dictionary_or_path) is str:
        with open(token_dictionary_or_path, 'rb') as fp:
            token_dictionary = pickle.load(fp)
    else:
        token_dictionary = token_dictionary_or_path

    with open(name2promoter_human_path, 'rb') as fp:
        name2promoter_human = pickle.load(fp)

    with open(name2promoter_mouse_path, 'rb') as fp:
        name2promoter_mouse = pickle.load(fp)

    with open(name2family_human_path, 'rb') as fp:
        name2family_human = pickle.load(fp)

    with open(name2family_mouse_path, 'rb') as fp:
        name2family_mouse = pickle.load(fp)
    
    with open(name2coexp_human_path, 'rb') as fp:
        name2coexp_human = pickle.load(fp)

    with open(name2coexp_mouse_path, 'rb') as fp:
        name2coexp_mouse = pickle.load(fp)

    with open(name2peca_human_path, 'rb') as fp:
        name2peca_human = pickle.load(fp)
    
    with open(name2peca_mouse_path, 'rb') as fp:
        name2peca_mouse = pickle.load(fp)

    with open(id2name_human_mouse_path, 'rb') as fp:
        id2name = pickle.load(fp)

    with open(homologous_gene_path, 'rb') as fp:
        homologous_gene_human2mouse = {v: k for k, v in pickle.load(fp).items()}

    token2promoter, token2coexp, token2family, token2peca = {}, {}, {}, {}
    for k, v in token_dictionary.items():
        if k not in ['<pad>', '<mask>']:
            if 'ENSG' in k:
                token2promoter[v] = name2promoter_human[id2name[k]]
                token2coexp[v] = name2coexp_human[id2name[k]]
                token2family[v] = name2family_human[id2name[k]]
                token2peca[v] = name2peca_human[id2name[k]]
            elif 'ENSMUSG' in k:
                token2promoter[v] = name2promoter_mouse[id2name[k]]             
                token2coexp[v] = name2coexp_mouse[id2name[k]]
                token2family[v] = name2family_mouse[id2name[k]]
                token2peca[v] = name2peca_mouse[id2name[k]]
            else:
                raise
        else:
            token2promoter[v] = np.zeros_like(name2promoter_human[list(name2promoter_human.keys())[0]])            
            token2coexp[v] = np.zeros_like(name2coexp_human[list(name2coexp_human.keys())[0]])
            token2family[v] = np.zeros_like(name2family_human[list(name2family_human.keys())[0]])
            token2peca[v] = np.zeros_like(name2peca_human[list(name2peca_human.keys())[0]])

    return (
        torch.as_tensor(np.stack([token2promoter[k] for k in sorted(token2promoter.keys())])),
        torch.as_tensor(np.stack([token2coexp[k] for k in sorted(token2coexp.keys())])),
        torch.as_tensor(np.stack([token2family[k] for k in sorted(token2family.keys())])),
        torch.as_tensor(np.stack([token2peca[k] for k in sorted(token2peca.keys())]), dtype=torch.float32),
        homologous_gene_human2mouse
    )


if __name__ == '__main__':
    x = load_prior_embedding(
        # NAME_TO_PROMOTER_HUMAN_PATH, NAME_TO_PROMOTER_MOUSE_PATH,
        # NAME_TO_COEXP_HUMAN_PATH, NAME_TO_COEXP_MOUSE_PATH,
        # NAME_TO_FAMILY_HUMAN_PATH, NAME_TO_FAMILY_MOUSE_PATH,
        # ID_TO_NAME_HUMAN_MOUSE_PATH,
        # TOKEN_DICTIONARY_PATH
    )
    print()
    pass