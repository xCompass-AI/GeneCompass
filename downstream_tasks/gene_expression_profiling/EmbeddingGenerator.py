def get_GeneCompass_cls_new_embedding(Path, dataset_path, checkpoint_path, get_emb=False,
                                      emb_file_path=None, prior_embedding_path=None):
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

    sys.path.append(Path[:-1])
    from genecompass import BertForMaskedLM
    from genecompass.utils import load_prior_embedding

    with open('./GRNgene.pkl', 'rb') as f:
        GRN_gene = pickle.load(f)

    with open('./human_mouse_tokens.pickle', 'rb') as f:
        id_token = pickle.load(f)

    with open('./Gene_id_name_dict.pickle', 'rb') as f:
        gene_id_to_name = pickle.load(f)

    # Build token to gene name mapping
    token_to_gene_name = {}
    for gene_id, token in id_token.items():
        if gene_id[:4] == 'ENSG' and gene_id in gene_id_to_name:
            token_to_gene_name[token] = gene_id_to_name[gene_id]

    data = load_from_disk(dataset_path)

    # Get target gene names from first cell
    gene_names = []
    for e in data[0]['input_ids']:
        if e in token_to_gene_name:
            gene_name = token_to_gene_name[e]
            if gene_name in GRN_gene and gene_name not in gene_names:
                gene_names.append(gene_name)

    # Load model and knowledge
    out = load_prior_embedding(prior_embedding_path)
    knowledges = {
        'promoter': out[0],
        'co_exp': out[1],
        'gene_family': out[2],
        'peca_grn': out[3],
        'homologous_gene_human2mouse': out[4]
    }

    model = BertForMaskedLM.from_pretrained(
        checkpoint_path,
        knowledges=knowledges,
        ignore_mismatched_sizes=True,
    ).to("cuda")
    model.eval()

    # Accumulate embeddings per gene
    gene_embeddings = {gene_name: [] for gene_name in gene_names}

    with torch.no_grad():
        for i in tqdm(range(len(data))):
            input_ids = torch.tensor(data[i]['input_ids']).unsqueeze(0).cuda()
            values = torch.tensor(data[i]['values']).unsqueeze(0).cuda()
            species = torch.tensor(data[i]['species']).unsqueeze(0).cuda()

            embeddings = model.bert.forward(input_ids=input_ids, values=values, species=species)[0]
            embeddings = embeddings[:, 1:, :].squeeze(0).cpu()

            tokens = data[i]['input_ids']

            for pos, token in enumerate(tokens):
                if token in token_to_gene_name:
                    gene_name = token_to_gene_name[token]
                    if gene_name in gene_embeddings:
                        gene_embeddings[gene_name].append(embeddings[pos])

    # Compute average embedding per gene
    gene_avg_embeddings = {}
    valid_gene_names = []

    for gene_name, emb_list in gene_embeddings.items():
        if len(emb_list) > 0:
            stacked_embs = torch.stack(emb_list, dim=0)
            gene_avg_embeddings[gene_name] = stacked_embs.mean(dim=0)
            valid_gene_names.append(gene_name)

    if len(valid_gene_names) == 0:
        raise ValueError("No valid gene embeddings found")

    # Reorder genes and embeddings
    ordered_gene_names = [gene for gene in gene_names if gene in valid_gene_names]
    ordered_embeddings = [gene_avg_embeddings[gene] for gene in ordered_gene_names]
    emb_tensor = torch.stack(ordered_embeddings, dim=0)

    # Compute cosine similarity matrix
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    n_genes = len(ordered_gene_names)
    final_similarity = np.zeros((n_genes, n_genes))

    for i in range(n_genes):
        for j in range(n_genes):
            sim = cos(emb_tensor[i].reshape(1, -1), emb_tensor[j].reshape(1, -1))
            final_similarity[i][j] = sim.item()

    # Binarize similarity matrix
    result = copy.deepcopy(final_similarity)
    if final_similarity.size > 500000:
        sorted_indices = np.argsort(final_similarity.ravel())[::-1]
        threshold_index = sorted_indices[499999]
        threshold_value = final_similarity.ravel()[threshold_index]
        result[final_similarity < threshold_value] = 0
        result[final_similarity >= threshold_value] = 1

    final_emb = emb_tensor.numpy()

    if get_emb and emb_file_path:
        np.save(emb_file_path, final_emb, allow_pickle=True)

    return ordered_gene_names, result, final_emb