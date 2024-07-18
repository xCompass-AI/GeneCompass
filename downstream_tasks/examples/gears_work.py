import numpy as np
import scanpy as sc
import sys
sys.path.append('../gears/gears_code')
from geares import PertData, GEARS

adata1 = sc.read_h5ad('/disk1/xCompass/data/gears/yzh_file.h5ad')

diaoyong_moxing_type_list=['id_only']

for diaoyong_moxing_type in diaoyong_moxing_type_list:

    print(diaoyong_moxing_type)

    # # get data
    pert_data = PertData('../data/gears/data')

    # # load dataset in paper: norman, adamson, dixit.
    # # pert_data.load(data_name = 'norman')

    # pert_data.new_data_process(dataset_name = 'norman_by_new_propress', adata = adata1)
    # # to load the processed data
    pert_data.load(data_path = '../data/gears/data/norman_by_new_propress')

    # # specify data split
    pert_data.prepare_split(split = 'simulation', seed = 1)
    # # get dataloader with batch size
    pert_data.get_dataloader(batch_size = 16, test_batch_size = 64)

    # # set up and train a model
    gears_model = GEARS(pert_data, device = 'cuda:0')
    gears_model.model_initialize(hidden_size = 64,damoxing_type=diaoyong_moxing_type)
    gears_model.train(epochs = 1)

    print('over')
    print(gears_model)

    config_filename='config_0617'+diaoyong_moxing_type+'_'+'.pkl'
    model_filename='model_0617'+diaoyong_moxing_type+'_'+'.pt'
    print(config_filename,model_filename)

    gears_model.save_model('model_folder',config_filename,model_filename)