from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import scanpy as sc
import networkx as nx
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from .data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from .utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts

class PertData:
    
    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True):
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            # 'BABP': {'GO:0004032', 'GO:0006693', 'GO:0044597', 'GO:0047718', 'GO:0051897', 'GO:0044598', 'GO:0016655', 'GO:0007586', 'GO:0005829', 'GO:0072582', 'GO:0031406', 'GO:0008202', 'GO:0047086', 'GO:0007186', 'GO:0032052', 'GO:0047115', 'GO:0016229', 'GO:0030855', 'GO:0044594', 'GO:0047023', 'GO:0047044', 'GO:0071395', 'GO:0071799', 'GO:0042448', 'GO:0018636', 'GO:0008284'
            # gene2go的数据格式如上，gene数量足够大
            self.gene2go = pickle.load(f)
    
    # 用于设置可以干扰的基因列表，并将它们包含在药物-基因网络中
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # 提供的Norman数据集是走的这里
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        # self.gene2go中提取essential_genes中包含的基因及其对应的GO（Gene Ontology）注释信息，并将提取的基因存储在self.pert_names中
        # 同时将基因映射到网络节点编号，存储在self.node_map_pert中
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}

    # 加载现有的数据集，并将其转换为可以用于 PyTorch Geometric (PyG) 的格式
    def load(self, data_name = None, data_path = None):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path
        """
        # 根据数据集名称或路径加载数据集
        if data_name in ['norman', 'adamson', 'dixit']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)            
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            # 保存到一个 AnnData 对象 self.adata 中
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either Norman/Adamson/Dixit "
                             "or a path to an h5ad file")
        
        # 设置扰动基因
        self.set_pert_genes()
        # 提醒用户在数据集中存在一些扰动基因不在 Gene Ontology (GO) 图中，因此无法进行扰动预测
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        # 少的几个基因
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        # 过滤掉不在 Gene Ontology (GO) 图中的扰动基因
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):

            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            
            
            print_sys("Creating pyg object for each cell in the data...")
            self.dataset_processed = self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")

    # 将新的数据集处理成可用于训练的格式
    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de = False):
        # 检查输入的 adata 对象是否包含必要的列（'condition'、'gene_name' 和 'cell_type'）
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = get_DE_genes(adata, skip_calc_de)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        print_sys("Creating pyg object for each cell in the data...")
        self.dataset_processed = self.create_dataset_file()
        print_sys("Saving new dataset pyg object at " + dataset_fname) 
        pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
        print_sys("Done!")
        
    # 用于准备训练集和测试集的划分，并将划分结果存储在self.set2conditions
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None):
        # 对应的几个划分数据集的方式
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split']
        # 如果划分方式不在规定好的范围之内，报错
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        self.train_gene_set_size = train_gene_set_size
        # 根据split、seed和train_gene_set_size参数生成划分文件的文件名，并检查是否已经存在该文件。
        # 如果存在，则从文件中加载划分结果；否则，根据split参数生成新的训练集和测试集划分，并将划分结果保存到文件中
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            # 根据split参数的不同，使用DataSplitter类生成不同的训练集和测试集划分
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)
            
            elif split == 'single':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)
            
            elif split == 'no_test':
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)
            
            elif split == 'no_split':          
                adata = self.adata
                adata.obs['split'] = 'test'
            
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    # 用于从数据集中创建训练、验证和测试数据加载器    
    def get_dataloader(self, batch_size, test_batch_size = None):
        # batch_size表示训练和验证数据加载器中的批量大小，test_batch_size表示测试数据加载器中的批量大小。
        # 如果未提供test_batch_size，则将其设置为batch_size的值。
        if test_batch_size is None:
            test_batch_size = batch_size

        # 使用AnnData对象中的基因名称列表self.adata.var.gene_name和节点映射字典self.node_map创建一个新的属性gene_names
        # 一个基因一个号
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs
        # 根据数据集的分割方式（self.split属性）创建一个或多个细胞图
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                # Norman直接走的这里
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                # 这里的set2condition就是划分这三个集合的字典，对应的value是列表。
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders

            # print('cell_graphs-train')
            # print(cell_graphs['train'][:40])
            # print(len(cell_graphs['train']))

            # print('===============================================')

            # print('cell_graphs-val')
            # print(cell_graphs['val'][:40])
            # print(len(cell_graphs['val']))

            train_loader = DataLoader(cell_graphs['train'], num_workers=12,
                                batch_size=batch_size, shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'], num_workers=12,
                                batch_size=batch_size, shuffle=True)

            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")
        #del self.dataset_processed # clean up some memory
    
    # 创建一个包含所有扰动组样本的数据集，并将其保存到一个字典中
    def create_dataset_file(self):
        dl = {}
        # 遍历所有扰动类型
        for p in tqdm(self.adata.obs['condition'].unique()):
            # 对于每个扰动类型，使用 create_cell_graph_dataset() 方法创建细胞图谱数据集
            cell_graph_dataset = self.create_cell_graph_dataset(self.adata, p, num_samples=1)
            dl[p] = cell_graph_dataset
        return dl
    
    def get_pert_idx(self, pert_category, adata_):
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')    
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    # Set up feature matrix and output
    # 用于创建 Gears 工具所需的输入数据格式的对象。设置输入特征矩阵和输出标签矩阵
    def create_cell_graph(self, X, y, de_idx, pert, emb ,pert_idx=None):
        
        # create_cell_graph 函数用于创建单个细胞图谱对象。
        # 该函数接受五个参数：X、y、de_idx、pert 和 pert_idx。
        # 其中，X 表示单个细胞的基因表达量数据，y 表示该细胞的标签（即分类标记），de_idx 表示不同类别之间差异表达的基因索引，pert 表示应用的扰动类型，pert_idx 表示应用扰动的基因索引（如果没有应用扰动，则设为 [-1]）。
        # 该函数的返回值是一个 Data 类型的对象，其中包含了所需的基因表达量数据、标签信息、扰动基因索引和扰动类型信息。

        #pert_feats = np.expand_dims(pert_feats, 0)
        #feature_mat = torch.Tensor(np.concatenate([X, pert_feats])).T
        feature_mat = torch.Tensor(X).T
        
        '''
        pert_feats = np.zeros(len(self.pert_names))
        if pert_idx is not None:
            for p in pert_idx:
                pert_feats[int(np.abs(p))] = 1
        pert_feats = torch.Tensor(pert_feats).T
        '''
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, emb=emb ,pert=pert)

    # 用于创建多个细胞图谱对象
    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs
        将多个细胞图谱对象组合成一个细胞图谱数据集
        """
        
        # 从 split_adata 中获取特定扰动类型的数据，并将其存储在 adata_ 中
        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        row_indices = np.where(split_adata.obs['condition'] == pert_category)[0]
        # print(row_indices)
        # print(len(row_indices))
        # 根据是否存在差异表达基因信息，获取用于划分不同类别的基因索引信息 de_idx
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []
        embb=[]
        # When considering a non-control perturbation
        # 根据扰动类型是否为控制组，获取应用扰动的基因索引信息 pert_idx
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            # 获取应用扰动的基因索引信息
            pert_idx = self.get_pert_idx(pert_category, adata_)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                # 从 de_genes 中获取差异表达最高的前 num_de_genes 个基因的索引信息。
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                # print('.................................................................................')
                # print(cell_z)
                # Use samples from control as basal expression
                # 在创建细胞图谱对象时，使用来自控制组的基因表达量数据作为基准表达
                # 如果扰动类型不是控制组，则对于每个细胞 cell_z，随机选择 num_samples 个控制组的样本作为基准表达，然后使用基准表达和 cell_z 中的基因表达量数据创建细胞图谱对象
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                # print(ctrl_samples.X.shape)
                for link,c in enumerate(ctrl_samples.X):
                    Xs.append(c)
                    ys.append(cell_z)
                    embb.append(ctrl_samples.obs['flag'][link])


        # When considering a control perturbation
        # 使用控制组作为参考组的情况
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for link,cell_z in enumerate(adata_.X):
                Xs.append(cell_z)
                ys.append(cell_z)
                embb.append(adata_.obs['flag'][link])


        # print('2')
        # Create cell graphs
        cell_graphs = []
        # print('······································································')
        # print(Xs)
        # print(ys)
        # print('······································································')
        for X, y,emb in zip(Xs, ys,embb):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, emb ,pert_idx))
        # print('3')
        return cell_graphs
