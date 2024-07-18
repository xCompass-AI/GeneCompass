import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch.nn as nn
from Model import VAE_EAD
from utils import evaluate, extractEdgesFromMatrix, newextractEdgesFromMatrix
import pickle
import copy

import sys
#sys.path.append("..")
import os
path1 = os.path.abspath(__file__)
#print(path1)
sys.path.append(path1[:-72])
from grn_inference import EmbeddingGenerator


Tensor = torch.cuda.FloatTensor
#Tensor = torch.FloatTensor

class non_celltype_GRN_model:
    def __init__(self, opt):
        self.opt = opt
        try:
            os.mkdir(opt.save_name)
        except:
            print('dir exist')

    def initalize_A(self, data):
        num_genes = data.shape[1]
        A = np.ones([num_genes, num_genes]) / (num_genes - 1) + (np.random.rand(num_genes * num_genes) * 0.0002).reshape(
            [num_genes, num_genes])
        for i in range(len(A)):
            A[i, i] = 0
        return A


    def init_data(self, Path, model, dataset_path, checkpoint_path, get_emb, prior_embedding_path = None):
        import random
        import numpy as np
        from numpy.random import MT19937
        from numpy.random import RandomState, SeedSequence
        import torch 
        import torch.backends.cudnn 
        import torch.cuda

        def set_determenistic_mode(SEED):
            torch.manual_seed(SEED)                       # Seed the RNG for all devices (both CPU and CUDA).
            random.seed(SEED)                             # Set python seed for custom operators.
            rs = RandomState(MT19937(SeedSequence(SEED))) # If any of the libraries or code rely on NumPy seed the global NumPy RNG.
            np.random.seed(SEED)             
            torch.cuda.manual_seed_all(SEED) 

        set_determenistic_mode(4)
        
        Path = self.opt.Path
        model = self.opt.model
        dataset_path = self.opt.dataset_path
        checkpoint_path = self.opt.checkpoint_path
        get_emb = self.opt.get_emb

        if get_emb == False:
            emb_file_path = None
        else:
            emb_file_path = self.opt.emb_file_path

        

        Ground_Truth = pd.read_csv(self.opt.net_file, header=0)
        data = sc.read(self.opt.data_file)
        gene_name = list(data.var_names)
        

        if model == 'GeneCompass_cls_new':
            
            embed_gene_index, embed, _ = EmbeddingGenerator.get_GeneCompass_cls_new_embedding(Path, dataset_path, checkpoint_path, get_emb, emb_file_path, prior_embedding_path)
            


        else:
            raise ValueError('mode can only be [GeneCompass_cls_new], please choose one of them')
        
        
        L = []
        for i in range(len(gene_name)):
            if gene_name[i] in embed_gene_index:
                L.append([i, embed_gene_index.index(gene_name[i])])
        
        
        

        data_values = data.X
        Dropout_Mask = (data_values != 0).astype(float)
        data_values = (data_values - data_values.mean(0)) / (data_values.std(0))
        data = pd.DataFrame(data_values, index=list(data.obs_names), columns=gene_name)
        TF = set(Ground_Truth['Gene1'])
        All_gene = set(Ground_Truth['Gene1']) | set(Ground_Truth['Gene2'])
        num_genes, num_nodes = data.shape[1], data.shape[0]
        Evaluate_Mask = np.zeros([num_genes, num_genes])
        
        
        

        

        TF_mask = np.zeros([num_genes, num_genes])
        for i, item in enumerate(data.columns):
            for j, item2 in enumerate(data.columns):
                if i == j:
                    continue
                if item2 in TF and item in All_gene:
                    Evaluate_Mask[i, j] = 1
                if item2 in TF:
                    TF_mask[i, j] = 1
        feat_train = torch.FloatTensor(data.values)

        train_data = TensorDataset(feat_train, torch.LongTensor(list(range(len(feat_train)))),
                                   torch.FloatTensor(Dropout_Mask))
        

        dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=1)
        truth_df = pd.DataFrame(np.zeros([num_genes, num_genes]), index=data.columns, columns=data.columns)
        for i in range(Ground_Truth.shape[0]):
            truth_df.loc[Ground_Truth.iloc[i, 1], Ground_Truth.iloc[i, 0]] = 1
        A_truth = truth_df.values
        idx_rec, idx_send = np.where(A_truth)
        truth_edges = set(zip(idx_send, idx_rec))
        return dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TF_mask, gene_name, L, embed

    def train_model(self):
        opt = self.opt
        Path = opt.Path
        model = opt.model
        dataset_path = opt.dataset_path
        checkpoint_path = opt.checkpoint_path
        get_emb = opt.get_emb
        prior_embedding_path = opt.prior_embedding_path

        
        dataloader, Evaluate_Mask, num_nodes, num_genes, data, truth_edges, TFmask2, gene_name, L, embed = self.init_data(Path, model, dataset_path, checkpoint_path, get_emb, prior_embedding_path)
        adj_A_init = self.initalize_A(data)
        
        vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float().cuda()
        #vae = VAE_EAD(adj_A_init, 1, opt.n_hidden, opt.K).float()

        optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
        optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
        best_Epr = 0
        vae.train()

        valid_ep = [-np.inf]
        for epoch in range(opt.n_epochs + 1):
            loss_all, mse_rec, loss_kl, data_ids, loss_tfs, loss_sparse = [], [], [], [], [], []
            if epoch % (opt.K1 + opt.K2) < opt.K1:
                vae.adj_A.requires_grad = False
            else:
                vae.adj_A.requires_grad = True
            for i, data_batch in enumerate(dataloader, 0):
                optimizer.zero_grad()
                inputs, data_id, dropout_mask = data_batch
                #print(inputs.size())
                #print(embed.size())
                inputs = Variable(inputs.type(Tensor))
                data_ids.append(data_id.cpu().detach().numpy())
                temperature = max(0.95 ** epoch, 0.5)
                loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=None,
                                                                           temperature=temperature, opt=opt)
                sparse_loss = opt.alpha * torch.mean(torch.abs(vae.adj_A))
                loss = loss + sparse_loss
                loss.backward()
                mse_rec.append(loss_rec.item())
                loss_all.append(loss.item())
                loss_kl.append(loss_gauss.item() + loss_cat.item())
                loss_sparse.append(sparse_loss.item())
                if epoch % (opt.K1 + opt.K2) < opt.K1:
                    optimizer.step()
                else:
                    optimizer2.step()
            scheduler.step()
            if epoch % (opt.K1 + opt.K2) >= opt.K1:
                
                newextractEdgesFromMatrix(vae.adj_A.cpu().detach().numpy(), gene_name, TFmask2, L, embed).to_csv(
                        opt.save_name + '/GRN_inference_result_{}.tsv'.format(epoch), sep='\t', index=False)