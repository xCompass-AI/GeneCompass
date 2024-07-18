import numpy as np
from scipy.stats import spearmanr
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description='Drug Repurposing')
parser.add_argument('--data_dir')
parser.add_argument('--patient_file')
parser.add_argument('--num_cell')
parser.add_argument('--top')

args = parser.parse_args()

data_dir = args.data_dir
patient_file = args.patient_file
num_cell = int(args.num_cell)
top = int(args.top)

"""
predicted_drug_sig_file_list = ['predicted_a375.csv', 'predicted_a549.csv', 'predicted_ha1e.csv', 'predicted_hela.csv',
                                'predicted_ht29.csv', 'predicted_mcf7.csv', 'predicted_pc3.csv', 'predicted_yapc.csv']
hq_drug_sig_file_list = ['hq_a375.csv', 'hq_a549.csv', 'hq_ha1e.csv', 'hq_hela.csv', 'hq_ht29.csv', 'hq_mcf7.csv',
                         'hq_pc3.csv', 'hq_yapc.csv']
"""

predicted_drug_sig_file_list = ['geneformer-5500w_human_4prior_ids_values_predict_a375.csv', 'geneformer-5500w_human_4prior_ids_values_predict_ha1e.csv', 'geneformer-5500w_human_4prior_ids_values_predict_hela.csv', 'geneformer-5500w_human_4prior_ids_values_predict_ht29.csv',
                                'geneformer-5500w_human_4prior_ids_values_predict_mcf7.csv', 'geneformer-5500w_human_4prior_ids_values_predict_pc3.csv', 'geneformer-5500w_human_4prior_ids_values_predict_yapc.csv']

hq_drug_sig_file_list = ['hq_a375.csv', 'hq_ha1e.csv', 'hq_hela.csv', 'hq_ht29.csv', 'hq_mcf7.csv',
                         'hq_pc3.csv', 'hq_yapc.csv']



l1000_gene_file = 'l1000_gene_list.csv'
predicted_drug_file = 'predicted_drug_list.csv'
if 'ngdc' in patient_file:
    sig_source = 'NGDC'
elif 'ncbi' in patient_file:
    sig_source = 'NCBI'

def get_drug_list(input_file):
    with open(data_dir + '/' + input_file) as f:
        drug_list = f.readline().strip().split(',')
    return drug_list


def get_l1000_gene(input_file):
    with open(data_dir + '/' + input_file) as f:
        l1000_gene = f.readline().strip().split(',')
    return l1000_gene


def read_drug_signature(input_file, idx):
    drug_sig = []
    with open(data_dir + '/' + input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            line = [float(i) for i in line]
            drug_sig.append(line)
    drug_sig = np.array(drug_sig)
    drug_sig = drug_sig[:, idx]
    return drug_sig


def read_hq_drug_signature(input_file, idx):
    drug_sig = []
    drug_id = []
    with open(data_dir + '/' + input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            drug_id.append(line[0])
            line = [float(i) for i in line[1:]]
            drug_sig.append(line)
    drug_sig = np.array(drug_sig)
    drug_sig = drug_sig[:, idx]
    return drug_sig, drug_id


def get_disease_signature(input_file, l1000_gene, sig_type):
    assert sig_type in ['NGDC', 'NCBI']
    idx = []
    patient_sig = []
    with open(data_dir + '/' + input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            if sig_type == 'NCBI':
                g = line[0].strip('"')
            elif sig_type == 'NGDC':
                g = line[7]
            if g in l1000_gene and 'NA' not in line:
                id = l1000_gene.index(g)
                idx.append(id)
                if sig_type == 'NCBI':
                    patient_sig.append(float(line[2]))
                elif sig_type == 'NGDC':
                    patient_sig.append(float(line[3]))
    return idx, patient_sig


def compute_correlation(drug_sig, disease_sig):
    drug_sig = np.array(drug_sig)
    disease_sig = np.array(disease_sig)
    #out = np.zeros((len(drug_sig)), dtype=np.float)
    out = np.zeros((len(drug_sig)), dtype=np.float64)
    for i, d1 in enumerate(drug_sig):
        out[i] = spearmanr(d1, disease_sig)[0]
    return out


def get_candidate_drug_id(correlation, drug_list, num_candidate):
    drug_list = np.array(drug_list)
    correlation_candidate_idx = np.argsort(correlation)[:num_candidate]
    return drug_list[correlation_candidate_idx].tolist()


l1000_gene = get_l1000_gene(l1000_gene_file)
drug_list = get_drug_list(predicted_drug_file)
idx, patient_sig = get_disease_signature(patient_file, l1000_gene, sig_source)
output_total_correlation = []
for f1, f2 in zip(predicted_drug_sig_file_list, hq_drug_sig_file_list):
    drug_sig = read_drug_signature(f1, idx)
    hq_drug_sig, hq_drug_list = read_hq_drug_signature(f2, idx)
    drug_list = drug_list + hq_drug_list
    drug_sig = np.concatenate((drug_sig, hq_drug_sig), axis=0)
    correlation = compute_correlation(drug_sig, patient_sig)
    output_correlation = get_candidate_drug_id(correlation, drug_list, num_candidate=top)
    output_total_correlation += output_correlation
print('Drug Repurposing Output:')
for drug, freq in Counter(output_total_correlation).items():
    if freq >= num_cell:
        print('%s appears in %d cell lines' % (drug, freq))
