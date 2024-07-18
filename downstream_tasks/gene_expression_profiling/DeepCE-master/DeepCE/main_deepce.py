import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from datetime import datetime
import torch
import numpy as np
import argparse
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/models')
# sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
from models import DeepCE
from utils import DataReader
from utils import rmse, correlation, precision_k
import pandas as pd

start_time = datetime.now()

parser = argparse.ArgumentParser(description='DeepCE Training')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size')
parser.add_argument('--max_epoch')
parser.add_argument('--emb_file')

args = parser.parse_args()

drug_file = args.drug_file
gene_file = args.gene_file
dropout = float(args.dropout)
gene_expression_file_train = args.train_file
gene_expression_file_dev = args.dev_file
gene_expression_file_test = args.test_file
batch_size = int(args.batch_size)
max_epoch = int(args.max_epoch)
emb_file = args.emb_file

# parameters initialization
drug_input_dim = {'atom': 62, 'bond': 6}
drug_embed_dim = 128
drug_target_embed_dim = 128
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
gene_embed_dim = 128
pert_type_emb_dim = 4
cell_id_emb_dim = 4
pert_idose_emb_dim = 4
hid_dim = 128
num_gene = 978
precision_degree = [10, 20, 50, 100]
loss_type = 'point_wise_mse'
intitializer = torch.nn.init.xavier_uniform_
filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
          "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
          "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}

# check cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

# device = torch.device("cpu")

data = DataReader(drug_file, gene_file, gene_expression_file_train, gene_expression_file_dev,
                  gene_expression_file_test, filter, device)
#sys.exit()
print('#Train: %d' % len(data.train_feature['drug']))
print('#Dev: %d' % len(data.dev_feature['drug']))
print('#Test: %d' % len(data.test_feature['drug']))


# model creation
model = DeepCE(drug_input_dim=drug_input_dim, drug_emb_dim=drug_embed_dim,
                      conv_size=conv_size, degree=degree, gene_input_dim=np.shape(data.gene)[1],
                      gene_emb_dim=gene_embed_dim, num_gene=np.shape(data.gene)[0], hid_dim=hid_dim, dropout=dropout,
                      loss_type=loss_type, device=device, initializer=intitializer,
                      pert_type_input_dim=len(filter['pert_type']), cell_id_input_dim=len(filter['cell_id']),
                      pert_idose_input_dim=len(filter['pert_idose']), pert_type_emb_dim=pert_type_emb_dim,
                      cell_id_emb_dim=cell_id_emb_dim, pert_idose_emb_dim=pert_idose_emb_dim,
                      use_pert_type=data.use_pert_type, use_cell_id=data.use_cell_id,
                      use_pert_idose=data.use_pert_idose)
model.to(device)
model = model.double()

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
best_dev_loss = float("inf")
best_dev_pearson = float("-inf")
pearson_list_dev = []
pearson_list_test = []
spearman_list_dev = []
spearman_list_test = []
rmse_list_dev = []
rmse_list_test = []
precisionk_list_dev = []
precisionk_list_test = []
pearson_raw_list = []
for epoch in range(max_epoch):
    print("Iteration %d:" % (epoch+1))
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):
        ft, lb = batch
        drug = ft['drug']
        mask = ft['mask']
        if data.use_pert_type:
            pert_type = ft['pert_type']
        else:
            pert_type = None
        if data.use_cell_id:
            cell_id = ft['cell_id']
        else:
            cell_id = None
        if data.use_pert_idose:
            pert_idose = ft['pert_idose']
        else:
            pert_idose = None
        optimizer.zero_grad()
        predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose, emb_file)
        loss = model.loss(lb, predict)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Train loss:')
    print(epoch_loss/(i+1))

    model.eval()

    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='dev', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose, emb_file)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            # print(predict_np.shape)
        print('Dev loss:')
        print(epoch_loss / (i + 1))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_dev.append(rmse_score)
        print('RMSE: %.4f' % rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_dev.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_dev.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_dev.append(precision)

        if best_dev_pearson < pearson:
            best_dev_pearson = pearson

    epoch_loss = 0
    lb_np = np.empty([0, num_gene])
    predict_np = np.empty([0, num_gene])
    cell_np = np.empty([0, 7])
    with torch.no_grad():
        for i, batch in enumerate(data.get_batch_data(dataset='test', batch_size=batch_size, shuffle=False)):
            ft, lb = batch
            #print(ft)
            drug = ft['drug']
            mask = ft['mask']
            if data.use_pert_type:
                pert_type = ft['pert_type']
            else:
                pert_type = None
            if data.use_cell_id:
                cell_id = ft['cell_id']
            else:
                cell_id = None
            if data.use_pert_idose:
                pert_idose = ft['pert_idose']
            else:
                pert_idose = None
            predict = model(drug, data.gene, mask, pert_type, cell_id, pert_idose, emb_file)
            loss = model.loss(lb, predict)
            epoch_loss += loss.item()
            lb_np = np.concatenate((lb_np, lb.cpu().numpy()), axis=0)
            predict_np = np.concatenate((predict_np, predict.cpu().numpy()), axis=0)
            cell_np = np.concatenate((cell_np, cell_id.cpu().numpy()), axis = 0)
        #print(predict_np.shape, cell_np.shape)
        cell = []
        for i in range(len(cell_np)):
            cell.append(filter['cell_id'][list(cell_np[i]).index(1)])
        #print(cell)

        """
        A375 = []
        HA1E = []
        HELA = []
        HT29 = []
        MCF7 = []
        PC3 = []
        YAPC = []

        for i in range(len(cell)):
            if cell[i] == 'A375':
                A375.append(i)
            elif cell[i] == 'HA1E':
                HA1E.append(i)
            elif cell[i] == 'HELA':
                HELA.append(i)
            elif cell[i] == 'HT29':
                HT29.append(i)
            elif cell[i] == 'MCF7':
                MCF7.append(i)
            elif cell[i] == 'PC3':
                PC3.append(i)
            elif cell[i] == 'YAPC':
                YAPC.append(i)
        #print(A375)
        pre_a375 = pd.DataFrame(predict_np[A375,:])
        #print(pre_a375)
        pre_ha1e = pd.DataFrame(predict_np[HA1E,:])
        pre_hela = pd.DataFrame(predict_np[HELA,:])
        pre_ht29 = pd.DataFrame(predict_np[HT29,:])
        pre_mcf7 = pd.DataFrame(predict_np[MCF7,:])
        pre_pc3 = pd.DataFrame(predict_np[PC3,:])
        pre_yapc = pd.DataFrame(predict_np[YAPC,:])

        pre_a375.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_a375.csv', index = False, header = False)
        pre_ha1e.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_ha1e.csv', index = False, header = False)
        pre_hela.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_hela.csv', index = False, header = False)
        pre_ht29.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_ht29.csv', index = False, header = False)
        pre_mcf7.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_mcf7.csv', index = False, header = False)
        pre_pc3.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_pc3.csv', index = False, header = False)
        pre_yapc.to_csv('/home/cnic/yangqm/projects/downstreamtasks/DeepCE-master/DeepCE/data/covid_data/geneformer-5500w_human_4prior_ids_values_predict_yapc.csv', index = False, header = False)
        """


        print('Test loss:')
        print(epoch_loss / (i + 1))
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_test.append(rmse_score)
        print('RMSE: %.7f' % rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_test.append(pearson)
        print('Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_test.append(spearman)
        print('Spearman\'s correlation: %.4f' % spearman)
        precision = []
        for k in precision_degree:
            precision_neg, precision_pos = precision_k(lb_np, predict_np, k)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            precision.append([precision_pos, precision_neg])
        precisionk_list_test.append(precision)

best_dev_epoch = np.argmax(pearson_list_dev)
print("Epoch %d got best Pearson's correlation on dev set: %.4f" % (best_dev_epoch + 1, pearson_list_dev[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on dev set: %.4f" % (best_dev_epoch + 1, spearman_list_dev[best_dev_epoch]))
print("Epoch %d got RMSE on dev set: %.4f" % (best_dev_epoch + 1, rmse_list_dev[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_dev[best_dev_epoch][-1][0],
                                                                  precisionk_list_dev[best_dev_epoch][-1][1]))

print("Epoch %d got Pearson's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, pearson_list_test[best_dev_epoch]))
print("Epoch %d got Spearman's correlation on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, spearman_list_test[best_dev_epoch]))
print("Epoch %d got RMSE on test set w.r.t dev set: %.4f" % (best_dev_epoch + 1, rmse_list_test[best_dev_epoch]))
print("Epoch %d got P@100 POS and NEG on test set w.r.t dev set: %.4f, %.4f" % (best_dev_epoch + 1,
                                                                  precisionk_list_test[best_dev_epoch][-1][0],
                                                                  precisionk_list_test[best_dev_epoch][-1][1]))

best_test_epoch = np.argmax(pearson_list_test)
print("Epoch %d got best Pearson's correlation on test set: %.4f" % (best_test_epoch + 1, pearson_list_test[best_test_epoch]))
print("Epoch %d got Spearman's correlation on test set: %.4f" % (best_test_epoch + 1, spearman_list_test[best_test_epoch]))
print("Epoch %d got RMSE on test set: %.4f" % (best_test_epoch + 1, rmse_list_test[best_test_epoch]))
print("Epoch %d got P@100 POS and NEG on test set: %.4f, %.4f" % (best_test_epoch + 1,
                                                                  precisionk_list_test[best_test_epoch][-1][0],
                                                                  precisionk_list_test[best_test_epoch][-1][1]))
end_time = datetime.now()
print(end_time - start_time)
print('Best Test RMSE:', min(rmse_list_test))