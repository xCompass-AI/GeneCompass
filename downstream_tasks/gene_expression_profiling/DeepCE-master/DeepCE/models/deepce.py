import torch
import torch.nn as nn
from .neural_fingerprint import NeuralFingerprint
from .drug_gene_attention import DrugGeneAttention
from .ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, \
    list_wise_ndcg
import numpy as np

class DeepCE(nn.Module):
    def __init__(self, drug_input_dim, drug_emb_dim, conv_size, degree, gene_input_dim, gene_emb_dim, num_gene,
                 hid_dim, dropout, loss_type, device, initializer=None, pert_type_input_dim=None,
                 cell_id_input_dim=None, pert_idose_input_dim=None,
                 pert_type_emb_dim=None, cell_id_emb_dim=None, pert_idose_emb_dim=None, use_pert_type=False,
                 use_cell_id=False, use_pert_idose=False):
        super(DeepCE, self).__init__()
        assert drug_emb_dim == gene_emb_dim, 'Embedding size mismatch'
        self.use_pert_type = use_pert_type
        self.use_cell_id = use_cell_id
        self.use_pert_idose = use_pert_idose
        self.drug_emb_dim = drug_emb_dim
        self.gene_emb_dim = gene_emb_dim
        self.drug_fp = NeuralFingerprint(drug_input_dim['atom'], drug_input_dim['bond'], conv_size, drug_emb_dim,
                                         degree, device)
        self.gene_embed = nn.Linear(gene_input_dim, gene_emb_dim)
        self.drug_gene_attn = DrugGeneAttention(gene_emb_dim, gene_emb_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=dropout, device=device)
        self.linear_dim = self.drug_emb_dim + self.gene_emb_dim
        if self.use_pert_type:
            self.pert_type_embed = nn.Linear(pert_type_input_dim, pert_type_emb_dim)
            self.linear_dim += pert_type_emb_dim
        if self.use_cell_id:
            self.cell_id_embed = nn.Linear(cell_id_input_dim, cell_id_emb_dim)
            self.linear_dim += cell_id_emb_dim
        if self.use_pert_idose:
            self.pert_idose_embed = nn.Linear(pert_idose_input_dim, pert_idose_emb_dim)
            self.linear_dim += pert_idose_emb_dim
        self.linear_1 = nn.Linear(self.linear_dim, hid_dim)
        self.linear_2 = nn.Linear(hid_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_gene = num_gene
        self.loss_type = loss_type
        self.initializer = initializer
        self.device = device
        self.init_weights()
        self.hid_dim = hid_dim

    def init_weights(self):
        if self.initializer is None:
            return
        for name, parameter in self.named_parameters():
            if 'drug_gene_attn' not in name:
                if parameter.dim() == 1:
                    nn.init.constant_(parameter, 0.)
                else:
                    self.initializer(parameter)

    def forward(self, input_drug, input_gene, mask, input_pert_type, input_cell_id, input_pert_idose, emb_file):
        # input_drug = {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}
        # gene_embed = [num_gene * gene_emb_dim]
        num_batch = input_drug['molecules'].batch_size
        drug_atom_embed = self.drug_fp(input_drug)
        # drug_atom_embed = [batch * num_node * drug_emb_dim]
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        # drug_embed = [batch * drug_emb_dim]
        drug_embed = drug_embed.unsqueeze(1)
        # drug_embed = [batch * 1 *drug_emb_dim]
        drug_embed = drug_embed.repeat(1, self.num_gene, 1)
        # drug_embed = [batch * num_gene * drug_emb_dim]
        gene_embed = self.gene_embed(input_gene)
        # gene_embed = [num_gene * gene_emb_dim]
        gene_embed = gene_embed.unsqueeze(0)
        # gene_embed = [1 * num_gene * gene_emb_dim]
        gene_embed = gene_embed.repeat(num_batch, 1, 1)
        # gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed, _ = self.drug_gene_attn(gene_embed, drug_atom_embed, None, mask)
        # drug_gene_embed = [batch * num_gene * gene_emb_dim]
        drug_gene_embed = torch.cat((drug_gene_embed, drug_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_emb_dim + gene_emb_dim)]

        
        # lm_embed = np.load('/home/cnic/yangqm/projects/downstreamtasks/DeepSEM-master/src/geneformer-xcompass-ict-128w_peca.npy')
        lm_embed = np.load(emb_file)
        lm_embed = torch.from_numpy(lm_embed)


        
        """
        file = open('/home/cnic/wangzj/new/CPA-fake/mouse_1_2.pkl', 'rb') #CPA-mouse-60w
        lm_embed = pickle.load(file)
        file.close()
        """

        lm_embed = lm_embed.permute(1,0)

        n = nn.Linear(lm_embed.size(1), self.num_gene)
        
        lm_embed = n(lm_embed).cuda()
        #lm_embed = n(lm_embed)

        lm_embed = lm_embed.permute(1,0)

        lm_embed = lm_embed.unsqueeze(0)
        lm_embed = lm_embed.repeat(num_batch, 1, 1)

        drug_gene_embed = torch.cat((drug_gene_embed, lm_embed), dim=2)

        
        if self.use_pert_type:
            pert_type_embed = self.pert_type_embed(input_pert_type)
            # pert_type_embed = [batch * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.unsqueeze(1)
            # pert_type_embed = [batch * 1 * pert_type_emb_dim]
            pert_type_embed = pert_type_embed.repeat(1, self.num_gene, 1)
            # pert_type_embed = [batch * num_gene * pert_type_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_type_embed), dim=2)
        if self.use_cell_id:
            cell_id_embed = self.cell_id_embed(input_cell_id)
            # cell_id_embed = [batch * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.unsqueeze(1)
            # cell_id_embed = [batch * 1 * cell_id_emb_dim]
            cell_id_embed = cell_id_embed.repeat(1, self.num_gene, 1)
            # cell_id_embed = [batch * num_gene * cell_id_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, cell_id_embed), dim=2)
        if self.use_pert_idose:
            pert_idose_embed = self.pert_idose_embed(input_pert_idose)
            # pert_idose_embed = [batch * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.unsqueeze(1)
            # pert_idose_embed = [batch * 1 * pert_idose_emb_dim]
            pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
            # pert_idose_embed = [batch * num_gene * pert_idose_emb_dim]
            drug_gene_embed = torch.cat((drug_gene_embed, pert_idose_embed), dim=2)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        drug_gene_embed = self.relu(drug_gene_embed)
        # drug_gene_embed = [batch * num_gene * (drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        
        #print(drug_gene_embed.dtype)
        #out = self.linear_1(drug_gene_embed) #未添加大模型信息时使用

        m = nn.Linear(drug_gene_embed.size(2), self.hid_dim).double().cuda()
        #m = nn.Linear(drug_gene_embed.size(2), self.hid_dim).double()

        out = m(drug_gene_embed)
        
        
        
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
        # out = [batch * num_gene]
        return out

    def loss(self, label, predict):
        if self.loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif self.loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif self.loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif self.loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif self.loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif self.loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % self.loss_type)
        return loss
