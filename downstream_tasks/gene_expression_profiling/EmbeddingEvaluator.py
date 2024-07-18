import random
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import torch 
import torch.backends.cudnn 
import torch.cuda
import os

def set_determenistic_mode(SEED):
    torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
    random.seed(SEED)                             # Set python seed for custom operators.
    rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
    np.random.seed(SEED)             
    torch.cuda.manual_seed_all(SEED) 

set_determenistic_mode(4)




def DeepSEM_result(path, data_file, net_file, save_name, model, dataset_path, checkpoint_path, n_epochs, get_emb = False, emb_file_path = None, prior_embedding_path = None):
    import subprocess
    import os
    import pandas as pd
    import numpy as np
    from sklearn.metrics import average_precision_score
    import sys
    
    path1 = os.path.abspath(__file__)

    subprocess.call('python {}/DeepSEM-master/main.py --Path {} --data_file {} --net_file {} --save_name {} --model {} --dataset_path {} --checkpoint_path {} --get_emb {} --n_epochs {} \
                    --emb_file_path {} --prior_embedding_path {}'.format(path1[:-22], path, data_file, net_file, save_name, model, dataset_path, checkpoint_path, get_emb, 
                                               n_epochs, emb_file_path, prior_embedding_path), shell=True)
    
    L = os.listdir(save_name)

    L.sort(key = lambda x:int(x[21:-4]))
    
    AUPRC= []
    for i in range(len(L)):
        output = pd.read_csv(os.path.join(save_name, L[i]),sep='\t')
        output['EdgeWeight'] = abs(output['EdgeWeight'])
        output = output.sort_values('EdgeWeight',ascending=False)
        
        label = pd.read_csv(net_file)

        Gene1s = set(label['Gene1'])
        Genes = set(label['Gene1'])| set(label['Gene2'])
        output = output[output['Gene1'].apply(lambda x: x in Gene1s)]
        output = output[output['Gene2'].apply(lambda x: x in Genes)]
        label_set = set(label['Gene1']+label['Gene2'])
        preds,labels,randoms = [] ,[],[]
        res_d = {}
        l = []
        p= []

        for item in (output.to_dict('records')):
            res_d[item['Gene1']+item['Gene2']] = item['EdgeWeight']
        for item in (set(label['Gene1'])):
            for item2 in  set(label['Gene1'])| set(label['Gene2']):
                if item+item2 in label_set:
                    l.append(1)
                else:
                    l.append(0)
                if item+ item2 in res_d:
                    p.append(res_d[item+item2])
                else:
                    p.append(-1)
        
        AUPRC.append(average_precision_score(l,p))
    
    print('AUPRC is: {}'.format(max(AUPRC)))



def CPA_result(adata_path, pretrained_cpa_model, llm_name, save_path, emb_file = None):
    import sys
    import cpa
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    import os

    

    dataname = 'GSM_new'
    
    adata = sc.read(adata_path)


    cpa_api = cpa.api.API(
    adata, 
    pretrained=pretrained_cpa_model
    )


    perts_anndata = cpa_api.get_drug_embeddings()
    covars_anndata = cpa_api.get_covars_embeddings('cell_type')
    cpa_api.compute_comb_emb(thrh=0)
    cpa_api.compute_uncertainty(
                    cov={'cell_type': 'A549'}, 
                    pert='Nutlin', 
                    dose='1.0'
                )
    df_reference = cpa_api.get_response_reference()


    
    reconstructed_response = cpa_api.get_response(n_points=10, emb_file = emb_file)

    df_reference = df_reference.replace('training_treated', 'train')

    
    llm_name = llm_name


    if os.path.exists(os.path.join(save_path, '{}-saved_plots'.format(llm_name))):
        print('dir exist')
    else:
        
        os.mkdir(os.path.join(save_path, '{}-saved_plots'.format(llm_name)))
    
    cpa_plots = cpa.plotting.CPAVisuals(cpa_api, fileprefix=None)  
    
    

    cpa_plots.plot_contvar_response(
        reconstructed_response, 
        df_ref=df_reference,
        response_name='MDM2',
        postfix='MDM2',
        title_name='Reconstructed dose response of MDM2',
        filename=save_path + '/{}-saved_plots/Reconstructed dose response of _MDM2_1'.format(llm_name))




    genes_control = cpa_api.datasets['training'].subset_condition(control=True).genes
    df_train = cpa_api.evaluate_r2(cpa_api.datasets['training'].subset_condition(control=False), genes_control, emb_file = emb_file)
    df_train['benchmark'] = 'CPA'

    df_train['model'] = '{}'.format(llm_name)


    genes_control = cpa_api.datasets['test'].subset_condition(control=True).genes
    df_ood = cpa_api.evaluate_r2(cpa_api.datasets['ood'], genes_control, emb_file = emb_file)
    df_ood['benchmark'] = 'CPA'

    df_ood['model'] = '{}'.format(llm_name)

    genes_control = cpa_api.datasets['test'].subset_condition(control=True).genes
    df_test = cpa_api.evaluate_r2(cpa_api.datasets['test'].subset_condition(control=False), genes_control, emb_file = emb_file)
    df_test['benchmark'] = 'CPA'

    df_test['model'] = '{}'.format(llm_name)

    df_test = cpa_api.evaluate_r2(cpa_api.datasets['test'].subset_condition(control=False), genes_control, emb_file = emb_file)
    df_test['benchmark'] = 'CPA'

    df_test['model'] = '{}'.format(llm_name)

    df_ood['split'] = 'ood'
    df_test['split'] ='test'
    df_train['split'] ='train'

    df_score = pd.concat([df_train, df_test, df_ood])
    df_score.round(2).sort_values(by=['condition', 'R2_mean', 'R2_mean_DE'], ascending=False)
    print(df_score.round(2).sort_values(by=['condition', 'R2_mean', 'R2_mean_DE'], ascending=False))
    

    df_score.round(2).sort_values(by=['condition', 'R2_mean', 'R2_mean_DE'], ascending=False).to_csv(path_or_buf=save_path + '/{}-saved_plots/{}.csv'.format(llm_name, dataname))



def DeepCE_result(drug_file, gene_file, train_file, dev_file, test_file, dropout, batch_size, max_epoch, emb_file, output_path):
    import subprocess
    import os

    subprocess.call('python ./DeepCE-master/DeepCE/main_deepce.py --drug_file {} \
                    --gene_file {} --train_file {} --dev_file {}  --test_file {} \
                    --dropout {} --batch_size {} --max_epoch {} --emb_file {} \
                    > {}'.format(drug_file, gene_file, train_file, dev_file, test_file, 
                                 dropout, batch_size, max_epoch, emb_file, output_path), shell=True)


