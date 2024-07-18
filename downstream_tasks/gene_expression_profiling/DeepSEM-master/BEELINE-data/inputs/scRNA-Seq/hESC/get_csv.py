import pandas as pd
import pickle

data = pd.read_csv('./ExpressionData.csv', index_col=0)
print(data)

#newdata =data.loc[['HLA-F', 'HLA-G', 'HLA-A', 'HLA-E', 'HLA-C', 'HLA-B', 'HLA-DRA', 'HLA-DRB5', 'HLA-DRB1', 'HLA-DQA1', 'HLA-DQB1', 'HLA-DQA2', 'HLA-DQB2', 'HLA-DOB', 'HLA-DMB', 'HLA-DMA', 'HLA-DOA', 'HLA-DPA1', 'HLA-DPB1']]
#print(newdata)

out_gene_name = open('/home/cnic02/yangqm/projects/Geneformer/gene_name.pkl', 'rb')
gene_name = pickle.load(out_gene_name)
out_gene_name.close()
print(gene_name)


new_geneformer_data = data.loc[gene_name]
print(new_geneformer_data)


new_geneformer_data.to_csv('./new_geneformer_ExpressionData.csv')