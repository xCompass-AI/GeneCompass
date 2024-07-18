from copy import deepcopy
import argparse
from time import time
import sys, os
import pickle

import scanpy as sc
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from .model import GEARS_Model
from .inference import evaluate, compute_metrics, deeper_analysis, \
                  non_dropout_analysis, compute_synergy_loss
from .utils import loss_fct, uncertainty_loss_fct, parse_any_pert, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS',
                 pred_scalar = False,
                 gi_predict = False):
        
        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gi_predict = gi_predict
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        if gi_predict:
            self.dict_filter = None
        else:
            self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                                self.adata.uns['non_zeros_gene_idx'].items() if
                                i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}

    def tunable_parameters(self):
        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, damoxing_type,
                         hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False, 
                         cell_fitness_pred = False,
                        ):
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'cell_fitness_pred': cell_fitness_pred,
                       'gene_list':self.gene_list,
                       'damoxing_type':damoxing_type,
                       'embedding_file_batch_dir_id_value_coexp':'/home/ict/yzh/new_emb_id_value_coexp64.pickle',
                       'embedding_file_batch_dir_id_value_emb':'/home/ict/yzh/new_emb_compass_id_value64.pickle',
                       'embedding_file_batch_dir_id_value_genefam':'/home/ict/yzh/new_emb_compass_id_value_genefam64.pickle',
                       'embedding_file_batch_dir_id_value_fourPrior_Att':'/home/ict/yzh/new_emb_compass_id_value_fourPrior_Att64.pickle',
                       'embedding_file_batch_dir_id_value_peca':'/home/ict/yzh/new_emb_id_value_peca64.pickle',
                       'embedding_file_batch_dir_id_only':'/disk1/xCompass/data/gears/id_only_emb.pickle',
                       'embedding_file_batch_dir_id_value_promoter':'/home/ict/yzh/new_emb_compass_id_value_promoter64.pickle',
                       'geneformer':'/disk1/ict00/yzh/0324_work/geneformer_data64.pickle',
                       'embedding_file_batch_dir_id_value_fourPrior_All':'/home/ict/yzh/new_emb_id_value_fourPrior_all64.pickle'
                      }
        #    'embedding_file_batch_dir':'/data/share/mu_work_256264.pkl'
        # /home/ict00/yangzhaohui/0728_work/geneformer_2.pickle
        # /home/ict00/yangzhaohui/0708_work/mu_work_256264.pkl
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)

            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path ,config_filename,model_filename):
        # with open(os.path.join(path, 'config_gears_final_big_model_0731.pkl'), 'rb') as f:
        with open(os.path.join(path, config_filename), 'rb') as f:
            config = pickle.load(f)

        del config['device'], config['num_genes'], config['num_perts'] ,config['gene_list'],config['embedding_file_batch_dir_geneformer'],config['embedding_file_batch_dir_big_model']
        # ,config['embedding_file_batch_dir_mouse']
        self.model_initialize(**config)
        self.config = config
        
        # state_dict = torch.load(os.path.join(path, 'model_gears_final_big_model_0731.pt'), map_location = torch.device('cpu'))
        state_dict = torch.load(os.path.join(path, model_filename), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path,config_filename,model_filename):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, config_filename), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, model_filename))
        

        # with open(os.path.join(path, 'config_gears_final_geneformer_out_ets2_and_cebpe_0731.pkl'), 'wb') as f:
        #     pickle.dump(self.config, f)
       
        # torch.save(self.best_model.state_dict(), os.path.join(path, 'model_gears_final_geneformer_out_ets2_and_cebpe_0731.pt'))
    

    # 用于对给定的单一或组合基因列表进行预测，并返回相应的转录组表达数据
    def predict(self, pert_list):
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        
        # 从 self.adata 中选择 condition 列中值为 'ctrl' 的所有样本存储在 self.ctrl_adata 变量
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        # 对于每个给定的扰动基因组合 pert_list，代码检查其是否在 self.pert_list 中
        # for pert in pert_list:
        #     for i in pert:
        #         if i not in self.pert_list:
        #             raise ValueError(i+ " is not in the perturbation graph. "
        #                                 "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        # 将 self.best_model 转换为 PyTorch 设备并设置为评估模式
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        # 创建一个空字典 results_pred，用于存储预测结果的平均值
        results_pred = {}  
        results_logvar_sum = {}
        
        from torch_geometric.data import DataLoader
        # 对于每个给定的扰动基因组合 pert，代码尝试从 self.saved_pred 中获取预测结果
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                # 如果结果已经存在，则跳过预测
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            # 调用函数创建 PyG 格式的数据集对象，并使用 PyTorch 数据加载器对其进行批处理
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)
            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
            
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        ## given a gene pair, return (1) transcriptome of A,B,A+B and (2) GI scores. 
        ## if uncertainty mode is on, also return uncertainty score.
        
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    # 这个函数的功能是绘制基因表达水平的扰动效应图。它接受两个参数：
    # query: 一个字符串，表示要绘制的实验条件，它是由多个实验条件组合而成，中间用"+"分隔。函数将计算这种条件下的基因与控制条件之间的差异表达（DE）。
    # save_file（可选）：一个字符串，表示要保存绘图的文件名。如果未提供，绘图将显示在屏幕上。
    def plot_perturbation(self, query, model_diaoyong ,save_file = None):
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.cbook import boxplot_stats


        print(query)
        # 自定义绘图样式，让绘制出来的图形更加美观易读。
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        # 从adata对象中提取必要的信息，包括基因表达数据、基因名和元信息
        adata = self.adata
        gene2idx = self.node_map
        # 将adata.obs中的"condition"和"condition_name"两列数据转换为一个字典
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        # 将基因的原始名称映射到它们的缩写名称
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        # 根据query和adata.uns中存储的数据，计算出指定实验条件下的差异表达（DE）基因的索引
        if query not in cond2name.keys():
            return [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

        de_idx = [gene2idx[gene_raw2id[i]] for i in
                  adata.uns['top_non_dropout_de_20'][cond2name[query]]]

        
        # print(de_idx)

        # genes是一个列表，存储了这些基因的名称
        genes = [gene_raw2id[i] for i in
                 adata.uns['top_non_dropout_de_20'][cond2name[query]]]  
        
        # print(genes)
        # print(len(genes))

        # # 输出表头
        # pre_data=[]
        # for i in range(len(adata.obs['flag'])):
        #     if adata.obs['control'][i]==1:
        #         pre_data.append(adata.obs['flag'][i])
        
        # print(len(pre_data))

        # raodong=[]

        # for i in range(len(adata.obs['condition'])):
        #     if adata.obs['condition'][i]=='ETS2+CEBPE':
        #         raodong.append(i)

        # print(len(raodong))

        # excel=[]
        # genename=list(adata.var['gene_name'])
        # for i in genes:
        #     dic={'gena_name':[],'ctrl_ex':[],'pre_ex':[],'gears_pre':[],'model_and_gears_pre':[]}
        #     num=genename.index(i)
        #     dic['gena_name']=i
        #     print(num)
        #     ex=[]
        #     for j in pre_data:
        #         ex.append(adata.X[j,num])
        #     dic['ctrl_ex']=ex
        #     excel.append(dic)
        #     print(dic)
        #     exit()

        # exit()
        # # 到这里


        # truth是一个二维数组，存储了指定实验条件下差异表达基因的表达水平(形状是相同扰动组合的细胞数，20（差异表达基因）)
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        # 拆分扰动组合
        query_ = [q for q in query.split('+') if q != 'ctrl']

        input_to_file=[]

        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values

        for i in range(100):
            pred = self.predict([query_])['_'.join(query_)][de_idx]
            pred = pred - ctrl_means
            input_to_file.append(pred)
        
        import pickle
        workpath='/data/home/ict00/GEARS-master/mouse_pred_data/'
        all_path=workpath+model_diaoyong+'_'+query+'.pkl'
        de_path=workpath+model_diaoyong+'_'+'de_'+query+'.pkl'
        print(all_path)
        print(de_path)
        with open(all_path, 'wb') as f:
            pickle.dump(input_to_file, f)

        with open(de_path, 'wb') as f:
            pickle.dump(de_idx, f)

        # ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values
        return

        pred = pred - ctrl_means
        truth = truth - ctrl_means

        plt.figure(figsize=[16.5,4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False,
                    medianprops = dict(linewidth=0))    

        for i in range(pred.shape[0]):
            _ = plt.scatter(i+1, pred[i], color='red')

        plt.axhline(0, linestyle="dashed", color = 'green')

        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation = 90)

        plt.ylabel("Change in Gene Expression over Control",labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()

        # 计算箱型图的统计值
        stats = boxplot_stats(truth, whis=[5, 95])

        # 计算每个散点到箱型图的距离
        distances = []
        for i in range(len(pred)):
            q1, median, q3 = stats[i]['whislo'], stats[i]['med'], stats[i]['whishi']
            # whisker_len = (q3 - q1) * 1.5
            # x = truth[:, i]
            y = pred[i]
            distance=abs(median-y)
            # if y >= q1 and y <= q3:
            #     distance = 0
            # elif y < q1:
            #     distance = abs(y - q1)
            # else:
            #     distance = abs(y - q3)
            # if distance > whisker_len:
            #     distance += 0.1 * (distance - whisker_len)  # 增加偏差值
            distances.append(distance)
        # 计算每个散点的离群点得分
        # max_distance = max(distances)
        # outlier_scores = [distance / max_distance for distance in distances]
        
        # print(outlier_scores)
        # print(distances)
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        # plt.show()

        return distances
    
    
    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        # print(self.gene_list)
        # for batch_index, batch_data in enumerate(train_loader):
        #     print("Batch index:", batch_index)
        #     print("Batch data:", batch_data.y)
        #     print("Batch data:", batch_data.y.size())
        #     print("Batch data:", batch_data.x.size())
        #     print("Batch data:", batch_data.x)
        #     exit()

        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):

                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      dict_filter = self.dict_filter,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  dict_filter = self.dict_filter,
                                  direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))
                
                # if step>1:
                #     print('先撤~')
                #     break


            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = ['frac_opposite_direction_top20_non_dropout',
                               'frac_sigma_below_1_non_dropout',
                               'mse_top20_de_non_dropout']
        
        if self.wandb:  
            for m in metrics:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            for m in metrics_non_dropout:
                self.wandb.log({'test_' + m: np.mean([j[m] for i,j in out_non_dropout.items() if m in j])})        

        if self.split == 'simulation':
            print_sys("Start doing subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in list(list(test_pert_res.values())[0].keys()):
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))

            ## deeper analysis
            subgroup_analysis = {}
            for name in subgroup['test_subgroup'].keys():
                subgroup_analysis[name] = {}
                for m in metrics:
                    subgroup_analysis[name][m] = []

                for m in metrics_non_dropout:
                    subgroup_analysis[name][m] = []

            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])

                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])

            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({'test_' + name + '_' + m: subgroup_analysis[name][m]})

                    print_sys('test_' + name + '_' + m + ': ' + str(subgroup_analysis[name][m]))
        print_sys('Done!')


