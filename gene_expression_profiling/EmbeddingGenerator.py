def get_GeneCompass_cls_new_embedding(Path, dataset_path, checkpoint_path, get_emb = False, emb_file_path = None, prior_embedding_path = None):

    from datasets import load_from_disk
    import torch
    from tqdm import tqdm
    import pickle

    from tqdm.notebook import trange
    import pandas as pd
    import torch.nn as nn
    import numpy as np
    import copy

    import sys
    #sys.path.append('/disk1/xCompass')
    sys.path.append(Path[:-1])
    from genecompass import BertForMaskedLM
    from genecompass.utils import load_prior_embedding

    
    GRN_gene_file = open('./GRNgene.pkl', 'rb')
    GRN_gene = pickle.load(GRN_gene_file)
    GRN_gene_file.close()
    

    

    file = open('./h_m_token2000W.pickle', 'rb')
    id_token = pickle.load(file)
    file.close()
    
    
    
    file = open('./Gene_id_name_dict.pickle', 'rb')
    gene = pickle.load(file)
    file.close()
    

    
    data = load_from_disk(dataset_path)

    gene_name = []
    gene_name_index = []
    for i in trange(len(data)):
        example_cell = data.select([i])
        
        for e in example_cell['input_ids'][0]:
            for i, v in id_token.items():
                
                if e == v and i[:4] == 'ENSG':
                    
                    
                    if i not in list(gene.keys()):
                        continue
                    else:
                        if gene[i] in GRN_gene:
                            gene_name.append(gene[i])
                            gene_name_index.append(example_cell['input_ids'][0].index(e))

        break
    

    # 生成embedding

    knowledges = dict()
    

    out = load_prior_embedding(
    # name2promoter_human_path, name2promoter_mouse_path, id2name_human_mouse_path,
    # token_dictionary
    # prior_embedding_path
    )
    knowledges['promoter'] = out[0]
    knowledges['co_exp'] = out[1]
    knowledges['gene_family'] = out[2]
    knowledges['peca_grn'] = out[3]
    knowledges['homologous_gene_human2mouse'] = out[4]

    

    model = BertForMaskedLM.from_pretrained(
        checkpoint_path,
        knowledges=knowledges,
        ignore_mismatched_sizes=True,
    ).to("cuda")

    model.eval()
    emb_list = []
    
    
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            input_id = torch.tensor(data[i]['input_ids']).unsqueeze(0).cuda()
            values = torch.tensor(data[i]['values']).unsqueeze(0).cuda()
            species = torch.tensor(data[i]['species']).unsqueeze(0).cuda()
            
            new_emb = model.bert.forward(input_ids=input_id, values= values, species=species)[0]
            new_emb = new_emb[:,1:,:].cpu()
            
            emb_list.append(new_emb)
    

    

    emb = torch.stack(emb_list, dim = 0).mean(dim = 0)
    emb = torch.squeeze(emb, dim = 0)

    # 计算cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    final = np.zeros((len(gene_name_index), len(gene_name_index)))
    for i in range(len(gene_name_index)):
        for j in range(len(gene_name_index)):
            
            final[i][j] = cos(emb[gene_name_index[i]].reshape(1, -1), emb[gene_name_index[j]].reshape(1, -1)).cpu().numpy()

    result = copy.deepcopy(final)
    
    index = np.argsort(final.ravel())[::-1][499999]
    pos = np.unravel_index(index, final.shape)
    result[final < final[pos]] = 0
    result[final >= final[pos]] = 1

    


    final_emb = []
    for i in range(len(gene_name_index)):
        final_emb.append(emb[gene_name_index[i]].reshape(1, -1).numpy())

    final_emb = np.concatenate((final_emb))
    

    if get_emb:
       
        np.save(emb_file_path, final_emb, allow_pickle=True)

    
    return gene_name, result, final_emb



