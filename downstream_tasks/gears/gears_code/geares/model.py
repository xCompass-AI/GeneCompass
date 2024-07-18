import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import pickle
from torch_geometric.nn import SGConv
import scanpy as sc
import numpy as np
import time
class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):
    """
    GEARS
    """

    def __init__(self, args):
        super(GEARS_Model, self).__init__()
        # print(args)
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.cell_fitness_pred = args['cell_fitness_pred']
        self.pert_emb_lambda = 0.2
        self.count=0


        if args['damoxing_type']=='id_value_coexp':
            with open(args['embedding_file_batch_dir_id_value_coexp'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))

        elif args['damoxing_type']=='id_value_emb':
            with open(args['embedding_file_batch_dir_id_value_emb'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_value_genefam':
            with open(args['embedding_file_batch_dir_id_value_genefam'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_value_peca':
            with open(args['embedding_file_batch_dir_id_value_peca'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_value_fourPrior_Att':
            with open(args['embedding_file_batch_dir_id_value_fourPrior_Att'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_value_fourPrior_All':
            with open(args['embedding_file_batch_dir_id_value_fourPrior_All'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_only':
            with open(args['embedding_file_batch_dir_id_only'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='id_value_promoter':
            with open(args['embedding_file_batch_dir_id_value_promoter'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))
        
        elif args['damoxing_type']=='geneformer':
            with open(args['geneformer'], 'rb') as f:
                self.concatenated_tensor = pickle.load(f)
                print(len(self.concatenated_tensor))

        print('loaded!')

        # self.concatenated_tensor = np.concatenate(input_data, axis=0)
        # self.concatenated_tensor = input_data

        adata1 = sc.read_h5ad('/disk1/xCompass/data/gears/yzh_file.h5ad')
        self.emb_ind=np.array(adata1.layers['new_layers'])
        # print(self.emb_ind)
        # print(self.emb_ind.shape)

        # adata1 = sc.read_h5ad(args['embedding_layers_dir'])
        # self.net_embeding=adata1.layers['my_layer']
        # with open('/home/ict00/yangzhaohui/0615_work/id_to_embedding.pickle', 'rb') as f:
        #     self.dict = pickle.load(f)
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup      
        # 将输入中的基因或者药物干预映射到一个低维的向量空间中      
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        # 一个多层感知机（MLP），用于将药物干预的位置嵌入和药物干预的特征嵌入融合起来。这个融合过程可以帮助模型更好地捕捉药物干预的影响。
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        


        # 这里很重要，是Scfoundation主要修改的地方
        '''
        我们将scFoundation与用于扰动预测任务的高级模型GEARS相结合。
        在最初的GEARS模型中,将基因共表达图与扰动信息相结合，以预测扰动后的基因表达。
        共表达图中的每个节点代表一个基因，最初随机嵌入，边缘连接共表达基因。此图不是特定于细胞的，而是在所有细胞中共享的。
        在我们的设置中,我们从scFoundation解码器获得了每个细胞的基因上下文嵌入,并将这些嵌入设置为图中的节点(方法)。
        这导致了用于预测扰动的细胞特异性基因共表达图(图第5A段)。
        '''
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        

        # 解码器：使用共享的多层感知机层（self.recovery_w）和基于全连接层的解码器（self.indv_w1 和 self.indv_b1）对特征向量进行解码，得到每个基因的重构结果。
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        # 跨基因的MLP层：使用跨基因的多层感知机层（self.cross_gene_state）对所有基因的特征向量进行计算和转换，得到所有基因的交叉表示。
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        
        # final gene specific decoder
        # 最终的基因特定解码器：使用基于全连接层的解码器（self.indv_w2 和 self.indv_b2）对每个基因的交叉表示进行解码，得到最终的基因特定的重构结果。
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        # 批量归一化层：使用批量归一化层（self.bn_emb，self.bn_pert_base 和 self.bn_pert_base_trans）对输入数据进行标准化，从而加速模型的训练和提高模型的性能。
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        # 不确定性模式：如果模型需要计算不确定性，使用一个多层感知机层（self.uncertainty_w）对特征向量进行计算，得到与每个基因对应的不确定性值
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        #if self.cell_fitness_pred:
        # 如果模型需要对细胞健康度进行预测，使用一个多层感知机层（self.cell_fitness_mlp）对所有基因的特征向量进行计算和转换，得到细胞健康度的预测结果
        self.cell_fitness_mlp = MLP([self.num_genes, hidden_size*2, hidden_size, 1], last_layer_act='linear')

    def forward(self, data):
        # 获取输入数据 data 中的 x 和 pert_idx。
        
        x, pert_idx = data.x, data.pert_idx

        # 如果 no_perturb 为 True，则将 x 展平并返回
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            # 获取批次中的图的数量 num_graphs
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            # 获取基因嵌入向量 emb，并通过 bn_emb 层进行批标准化。然后，通过 emb_trans 层对 emb 进行变换，得到 base_emb

            # 这里的input为[0,1,2,....,5043,5044,0,1,2,3...]循环batch_size遍
            input=torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device'])
            # print(input[0])
            # print(input[5045])
            emb = self.gene_emb(input)

            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)   
            # print(base_emb.size())
            # 获取位置嵌入向量 pos_emb，并在 G_coexpress 和 G_coexpress_weight 的帮助下通过一系列 layers_emb_pos 层进行变换，最终得到 pos_emb。
            # 然后，将 pos_emb 与 base_emb 相加，并通过 emb_trans_v2 层进行变换，得到 base_emb。
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()


            this_batch=[]

            for i in data.emb:
                this_batch.append(self.concatenated_tensor[i])
            this_batch = np.array(this_batch)

            this_batchs = np.concatenate(this_batch, axis=0)
            this_batchs = torch.from_numpy(this_batchs).float().cuda()

            link=np.array(list(range(len(base_emb))))
            yu=np.array(link//3040)

            shang=np.array(link%3040)
            data.emb=np.array(data.emb)

            row=np.array(data.emb[yu])
            link_final=2048*(yu)+self.emb_ind[row,shang]
            base_emb=this_batchs[link_final]


            # get perturbation index and embeddings
            # 获取进行基因表达干扰的样本的索引和干扰嵌入向量
            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            # augment global perturbation embedding with GNN
            # 这段代码使用 GNN（图神经网络）来增强全局干扰嵌入
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()
            

            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1
            # print(out.size())

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            if self.cell_fitness_pred:
                return torch.stack(out), self.cell_fitness_mlp(torch.stack(out))
            # exit()
            return torch.stack(out)
        


'''
这段代码定义了一个名为 GEARS_Model 的 PyTorch 模块，它实现了一个针对基因表达数据分析的图神经网络（GNN）。
GNN 的输入是一个表示基因表达数据的图，其中节点对应于基因，边表示基因之间的共表达关系。图还可以包括干扰信息，其中某些基因在某些样本中被干扰。

GEARS_Model 类扩展了 PyTorch 的 nn.Module 类，并在其构造函数中定义了各种层和参数。
这些包括为基因和干扰定义的嵌入层，用于建模共表达和干扰相似性关系的 GNN 层，用于解码基因表达值的 MLP 层以及用于正则化的批标准化层。

GEARS_Model 类的 forward 方法以表示一批图数据的 PyTorch Data 对象为输入。
该方法首先使用构造函数中定义的嵌入层提取基因和干扰的基本嵌入。
然后，使用 GNN 层增强了基因嵌入的共表达信息，并使用另一个 GNN 层增强了干扰嵌入的干扰相似性信息。然后，使用 MLP 层将增强的基因和干扰嵌入组合起来，以预测基因表达值。

如果构造函数中的 uncertainty 标志设置为 True，则该方法还会输出每个基因表达值的预测对数方差向量，这可以用于模拟预测中的不确定性。
如果 cell_fitness_pred 标志设置为 True，该方法还会为批中的每个样本输出预测的细胞健康度值。
'''