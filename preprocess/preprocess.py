import pandas as pd  
import anndata 
import scanpy as sc
from scipy.stats import zscore
import numpy as np
import pickle 
import tqdm
from datasets import Dataset,Features,Sequence,Value
import os

# 从文件列表中读取指定物种线粒体基因、蛋白质基因、miRNA基因的基因名
# def get_gene_list(species: str, file_list: list[str]):
def get_gene_list(species, file_list):
    #species:human file_list:
    take_list = []
    protein_list = []
    miRNA_list = []
    mitochondria_list = []
    for i in file_list:
        if (i.split('/')[-1].split('_')[0] == species):
            take_list.append(i)
    for i in take_list:
        if (i.split('.')[-1] == "txt"):
            with open(i, "r") as f:
                name_temp = str(f.name.split('/')[-1].split('_')[1])
                if name_temp == 'protein':
                    for line in f:
                        first_word = line.split()[0]
                        protein_list.append(first_word)
                elif name_temp == 'miRNA.txt':
                    for line in f:
                        first_word = line.split()[0]
                        miRNA_list.append(first_word)
        elif i.split('.')[-1] == "xlsx":
            df = pd.read_excel(i)
            mitochondria_list = df.iloc[:,1].to_list()

    return protein_list, miRNA_list, mitochondria_list

# 根据gene_id过滤基因
def gene_id_filter(adata, gene_list) -> anndata:

    mask = adata.var.index.isin(gene_list)
    adata = adata[:, mask]
    return adata

def id_name_match(protein_list, miRNA_list):
    p_l = []
    for i in protein_list:
        if dict1.get(i) != None:
            p_l.append(dict1.get(i))

    m_l = []
    for i in miRNA_list:
        if dict1.get(i) != None:
            m_l.append(dict1.get(i))

    return p_l,m_l

def id_name_match1(name_list):
    n_l = []
    for i in name_list:
        if dict1.get(i) != None:
            n_l.append(dict1.get(i))
        else:
            n_l.append('delete')
    
    return n_l

def id_token_match(name_list):
    m_l = []
    for i in name_list:
        if token_dict.get(i) != None:
            m_l.append(i)
        else:
            m_l.append('delete')
    return m_l

# 过滤基因总量或线粒体基因总量在所有细胞平均值三个标准差之外的的细胞
def normal_filter(adata, mito_list) -> anndata:
    total_counts = adata.X.sum(axis=1) # 计算每一行之和
    # adata.X : 50000 * 55655 行表示细胞，列表示基因
    # adata.X存储的是矩阵信息，它的结构虽然是一个数组，但是没有行名和列名信息。矩阵的行名信息存储在adata.obs中，列名信息存储在adata.var中。
    total_counts = np.array(total_counts).squeeze() #len = 50000

    idx = total_counts > 0 #创建一个布尔索引数组
    adata = adata[idx, :] #使用布尔索引idx筛选adata的行，只保留idx中对应位置为True的行数据
    total_counts = total_counts[idx] #只保留idx中对应位置为True的元素
    gene_name = adata.var.gene_symbols if 'gene_symbols' in adata.var else adata.var.feature_name
    gene_name = gene_name.tolist()
    index = [element in mito_list for element in gene_name] #根据mito_list中的元素是否存在于gene_name中，创建一个布尔索引列表index
    mito_adata = adata[:, index] #使用布尔索引index筛选adata的列，只保留index中对应位置为True的列数据

    # 计算每个细胞的线粒体基因表达之和
    mito_counts = mito_adata.X.sum(axis=1)
    mito_counts = np.array(mito_counts).squeeze()
    mito_percentage = mito_counts / total_counts

    # 将每个变量的值进行标准化，计算z-score
    total_counts_zscore = zscore(total_counts) #计算总基因表达量的z-score，即对总基因表达量进行标准化。
    mito_percentage_zscore = zscore(mito_percentage) #计算线粒体基因表达百分比的z-score，即对线粒体基因表达百分比进行标准化

    # 筛选出符合条件的细胞
    keep_cells = ((total_counts_zscore > -3) & (total_counts_zscore < 3) & (mito_percentage_zscore > -3) & (
                mito_percentage_zscore < 3))

    # 仅保留符合条件的细胞
    adata = adata[keep_cells, :] #筛选行
    return adata

# 过滤编码蛋白质或miRNA的基因数少于7个的细胞
def gene_number_filter(adata, gene_list):
    indices = [k in gene_list for k in adata.var.index] 
    f_adata = adata[:, indices]
    data = f_adata.X.toarray()
    mask = np.count_nonzero(data, axis=1) > 6
    #计算data数组每行中非零元素的个数，并创建一个布尔索引数组mask，其中对应行的非零元素个数大于6的为True，小于等于6的为False
    return adata[mask]

def tokenize_cell(gene_vector, gene_list, token_dict):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    nonzero_mask = np.nonzero(gene_vector)[0] #返回非零位置索引
    sorted_indices = np.argsort(-gene_vector[nonzero_mask]) #对gene_vector中非零元素进行降序排序，并将排序后的索引存储在sorted_indices中
    gene_list = np.array(gene_list)[nonzero_mask][sorted_indices] #从gene_list中选择相应的基因，并按照排序顺序存储在gene_list中

    f_token = [token_dict[gene] for gene in gene_list]
    value = gene_vector[nonzero_mask][sorted_indices]
    return f_token, value.tolist()

def Normalized(adata,dict_path,Parmerters):

    matrix_a = adata.X
    with open(dict_path, 'rb') as f:
        dict_gene = pickle.load(f)
    
    # list_t存储非零中值
    gene_list_t = adata.var.index.to_list()
    gene_nonzero_t = []
    for i in gene_list_t:
        if i in dict_gene:
            gene_nonzero_t.append(dict_gene[i])
        else:
            gene_nonzero_t.append(1)
    gene_nonzero_t = np.array(gene_nonzero_t)

    # 中值dict_gene: {...ENSG00000185775': 1.4178609997034073, 'ENSG00000237675': 0.8293528705835342, 'ENSG00000206102': 0.44774532318115234}
    
    # 计算每个细胞的非零值的和
    # 计算matrix_a矩阵每行的非零值之和，并将结果存储在per_cell_nonzero_sum变量中
    per_cell_nonzero_sum = np.sum(matrix_a, axis=1) 
    # 计算matrix_a矩阵每行的非零值个数，并将结果存储在nonzero_count变量中
    nonzero_count = np.count_nonzero(matrix_a, axis=1)
    # 将per_cell_nonzero_sum中对应nonzero_count为0的元素置为0，对应行中非零值个数为0的行的非零值之和为0
    per_cell_nonzero_sum[nonzero_count == 0] = 0
    #subview_norm_array = np.nan_to_num(matrix_a[:, :].T / per_cell_nonzero_sum * Parmerters / gene_nonzero_t[:, None])

    #  计算矩阵matrix_a按列进行转置后除以基因中值，并使用np.nan_to_num函数将结果中的NaN值替换为0
    subview_norm_array = np.nan_to_num(matrix_a[:, :].T / gene_nonzero_t[:, None])
    subview_norm_array = np.array(subview_norm_array.T)
    adata.X = subview_norm_array
    return adata

def log1p(adata):
    sc.pp.log1p(adata, base=2) #底数为2
    return adata

def rank_value(adata, token):
    
    input_ids = np.zeros((len(adata.X), 2048))  # Initialize input_ids as a 2D array filled with zeros
    values = np.zeros((len(adata.X), 2048))  # Initialize values as a 2D array filled with zeros
    length = []

    gene_id = adata.var.index.to_list()

    # 按行遍历，一行为一个细胞
    for index, i in enumerate(tqdm.tqdm(adata.X)):
        i = np.squeeze(np.asarray(i))
        tokenizen, value = tokenize_cell(i, gene_id, token)
        # 处理2048截断
        if len(tokenizen) > 2048:
            input_ids[index] = tokenizen[:2048]
            values[index] = value[:2048]
            length.append(2048)
        else:
            input_ids[index, :len(tokenizen)] = tokenizen
            values[index, :len(value)] = value
            input_ids[index, len(tokenizen):] = 0  # Fill remaining elements with zeros
            values[index, len(value):] = 0  # Fill remaining elements with zeros
            length.append(len(tokenizen))

    return input_ids,length,values

def transfor_out(specices_str,length,input_ids,values):
    if specices_str == 'human':
        specices_int = 0
    elif specices_str =='mouse':
        specices_int = 1
    specices_int = [specices_int] * adata.X.shape[0]
    specices_int = [[x] for x in specices_int]
    length = [[x] for x in length]

    data_out = {'input_ids': input_ids,'values':values,'length': length,'species': specices_int}

    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'values': Sequence(feature=Value(dtype='float32')),
        'length': Sequence(feature=Value(dtype='int16')),
        'species': Sequence(feature=Value(dtype='int16')),
    })
    dataset = Dataset.from_dict(data_out, features=features)
    return dataset

def save_disk(patch_str,dataset,length):
    dataset.save_to_disk(patch_str)
    sorted_list = sorted(length)
    out_path = patch_str + '/sorted_length.pickle'
    with open(out_path, 'wb') as f:
        pickle.dump(sorted_list, f)

# dict_path = 'scdata/human_medium.pickle' #中值字典路径，需要改
dict_path = 'scdata/dict/human_gene_median_after_filter.pickle' #中值字典路径

# gene_token_path = 'scdata/h&m_token2000W.pickle' #token路径，需要改
gene_token_path = 'scdata/dict/human_mouse_tokens.pickle'

f_list = ["scdata/mouse_protein_coding.txt", "scdata/human_protein_coding.txt",
               "scdata/mouse_miRNA.txt", "scdata/human_miRNA.txt",
               "scdata/human_mitochondria.xlsx", "scdata/mouse_mitochondria.xlsx"]
specices_str = 'human'
gene_id_name_path = 'scdata/Gene_id_name_dict1.pickle'
gene_id_path = 'scdata/gene_id_hpromoter.pickle' #(promoter gen2vec 与protein和gene2vec交集)还需改
dir_path = 'scdata/human_r8'
out_path = 'scdata/output/'

with open(gene_id_name_path, 'rb') as f:
    dict1 = pickle.load(f)

with open(gene_token_path, 'rb') as f:
    token_dict = pickle.load(f)

#1. 取list:均为名字list
protein_list, miRNA_list, mitochondria_list = get_gene_list(species=specices_str, file_list=f_list)

#2. name_id映射
#采用Gene_id_name_dict1.pickle
protein_list, miRNA_list= id_name_match(protein_list, miRNA_list)

with open(gene_id_path,'rb') as f:
    gene_id = pickle.load(f)

patch_id = 1

print("Started:")
df = pd.read_csv("SRR17066581_count.csv")
# df = pd.read_csv("scdata/test/GSM6537936_count.csv")

df = df.rename(columns={
    'Unnamed: 0': 'cell_id'
}).set_index('cell_id').T

adata = sc.AnnData(df)
# original_df = df
# print(len(adata.var))

# 基因名要进行映射
#1.id_name translate
print("1.name_id translate")
gene_id_l = id_name_match1(name_list = adata.var.index.to_list())
adata.var['gene_symbols'] = adata.var.index
adata.var.index = gene_id_l
print(len(adata.obs))
print(len(adata.var))
adata = adata[:, ~(adata.var.index == "delete")]
print(len(adata.obs))
print(len(adata.var))

#2.过滤基因总量或线粒体基因总量在所有细胞平均值三个标准差之外的的细胞
print("2.filter cells by gene counts")
adata = normal_filter(adata, mitochondria_list)

print(len(adata.obs))
print(len(adata.var))

#3.取指定gene_id 根据gene_id过滤基因
print("3.gene_id match")
adata = gene_id_filter(adata, gene_id)
print(len(adata.obs))
print(len(adata.var))

#4.filter蛋白质和mirna 过滤编码蛋白质或miRNA的基因数少于7个的细胞
print("4.filter cells by gene number")
adata = gene_number_filter(adata, protein_list + miRNA_list)
print(len(adata.obs))
print(len(adata.var))

#4+.filter不在token字典里面的基因
print("4#.filter cells by gene token")
gene_id_name_m = id_token_match(name_list = adata.var.index.to_list())
adata.var['gene_symbols'] = adata.var.index
adata.var.index = gene_id_name_m
print(len(adata.obs))
print(len(adata.var))
adata = adata[:, ~(adata.var.index == "delete")]

#5.Normalized
print("5.Normalized")
adata = Normalized(adata,dict_path,Parmerters=1e4)
print(len(adata.obs))
print(len(adata.var))

#6.log1p
print("6.log1p")
adata = log1p(adata)
print(len(adata.obs))
print(len(adata.var))

#7.Rank
print("7.rank value")
input_ids,length,values = rank_value(adata,token_dict)

#8.输出Hungface_dataset
print("8.dataset transfor")
datasets = transfor_out(specices_str,length,input_ids,values)

#9.存储
patch = 'patch'+ str(patch_id)
path = out_path+patch
if not os.path.exists(path):
    os.makedirs(path)
print("9.save disk to : {}".format(patch))
save_disk(path,datasets,length)