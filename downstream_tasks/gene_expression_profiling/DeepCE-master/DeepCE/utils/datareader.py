from .data_utils import *

seed = 343
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)


class DataReader(object):
    def __init__(self, drug_file, gene_file, data_file_train, data_file_dev, data_file_test,
                 filter, device):
        self.device = device
        self.drug, self.drug_dim = read_drug_string(drug_file)
        self.gene = read_gene(gene_file, self.device)
        feature_train, label_train = read_data(data_file_train, filter)
        feature_dev, label_dev = read_data(data_file_dev, filter)
        feature_test, label_test = read_data(data_file_test, filter)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, \
        self.dev_label, self.test_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, self.drug, self.device)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = convert_smile_to_feature(feature['drug'][excerpt], self.device)
            output['mask'] = create_mask_feature(output['drug'], self.device)
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt]
