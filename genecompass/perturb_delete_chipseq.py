# imports
import itertools as it
import logging
import pickle
import seaborn as sns
import tqdm
from genecompass.modeling_bert import BertForMaskedLM
import numpy as np

from genecompass.utils import load_prior_embedding

sns.set()
import torch
from collections import defaultdict
from datasets import load_from_disk,Features,Sequence,Value,Dataset
from tqdm.notebook import trange

import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
TOKEN_DICTIONARY_FILE = '../prior_knowledge/human_mouse_tokens.pickle'
out = load_prior_embedding(token_dictionary_or_path=TOKEN_DICTIONARY_FILE)

knowledges = dict()
knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]
def quant_layers(model):
    layer_nums = []
    for name, parameter in model.named_parameters():
        if "layer" in name:
            layer_nums += [name.split("layer.")[1].split(".")[0]]
    return int(max(layer_nums)) + 1


def flatten_list(megalist):
    return [item for sublist in megalist for item in sublist]


def forward_pass_single_cell(model, example_cell, layer_to_quant):
    example_cell.set_format(type="torch")
    input_data = example_cell["input_ids"][0].unsqueeze(0)
    values = example_cell["values"][0].unsqueeze(0)
    species = torch.zeros(input_data.size()[0], 1).long()
    with torch.no_grad():
        outputs = model.bert.forward(input_ids=input_data.long().to('cuda:1'), values=values.to('cuda:1'),species=species.to('cuda:1'))[0]
    emb = outputs[:,1:,:]
    emb = emb.squeeze(dim=0)
    del outputs
    return emb


def perturb_emb_by_index(emb, indices):
    mask = torch.ones(emb.numel(), dtype=torch.bool)
    mask[indices] = False
    return emb[mask]


def delete_index(example):
    indexes = example["perturb_index"]
    if len(indexes) > 1 and type(indexes[0]) is list:
        indexes = flatten_list(indexes)
    for index in sorted(indexes, reverse=True):
        del_temp = example['input_ids']
        del del_temp[index]
        example.data['input_ids'] = del_temp
    return example


def overexpress_index(example):
    perturb_parameters = 10000
    indexes = example["perturb_index"]
    if len(indexes) > 1:
        indexes = flatten_list(indexes)
    for index in sorted(indexes, reverse=True):
        if(example.data['input_ids'] == None):
            continue
        perturb_x = example.data['input_ids']
        perturb_index = perturb_x[index][0]
        perturb_x[index][-1] = perturb_x[index][-1] * perturb_parameters
        sorted_list = sorted(perturb_x, key=lambda x: x[1], reverse=True)
        example.data['input_ids'] = sorted_list
        for i, data in enumerate(sorted_list):
            if data[0] == perturb_index:
                example.data['perturb_index'] = i
    return example


def make_perturbation_batch(example_cell,
                            perturb_type,
                            tokens_to_perturb,
                            anchor_token,
                            combo_lvl,
                            num_proc):
    if tokens_to_perturb == "all":
        if perturb_type in ["overexpress", "activate"]:
            range_start = 1
        elif perturb_type in ["delete", "inhibit"]:
            range_start = 0
        indices_to_perturb = [[i] for i in range(range_start, example_cell["length"][0][0])]
    elif combo_lvl > 0 and (anchor_token is not None):
        example_input_ids = example_cell["input_ids "][0]
        anchor_index = example_input_ids.index(anchor_token[0])
        indices_to_perturb = [sorted([anchor_index, i]) if i != anchor_index else None for i in
                              range(example_cell["length"][0])]
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]
    else:
        example_input_ids = example_cell["input_ids"][0]
        indices_to_perturb = [[example_input_ids.index(token)] if token in example_input_ids else None for token in
                              tokens_to_perturb]
        indices_to_perturb = [item for item in indices_to_perturb if item is not None]

    # create all permutations of combo_lvl of modifiers from tokens_to_perturb
    if combo_lvl > 0 and (anchor_token is None):
        if tokens_to_perturb != "all":
            if len(tokens_to_perturb) == combo_lvl + 1:
                indices_to_perturb = [list(x) for x in it.combinations(indices_to_perturb, combo_lvl + 1)]
        else:
            all_indices = [[i] for i in range(example_cell["length"][0])]
            all_indices = [index for index in all_indices if index not in indices_to_perturb]
            indices_to_perturb = [[[j for i in indices_to_perturb for j in i], x] for x in all_indices]
    if len(indices_to_perturb) > 1:
        indices_to_perturb[0].append(indices_to_perturb[1][0])
        indices_to_perturb = [indices_to_perturb[0]]
    length = len(indices_to_perturb)
    input_ids = example_cell["input_ids"][0]
    values = example_cell["values"][0]
    input = [np.stack([np.array(input_ids), np.array(values)], axis=1).tolist()]
    perturbation_dataset = Dataset.from_dict(
        {"input_ids": input,"perturb_index": indices_to_perturb})
    if length < 400:
        num_proc_i = 1
    else:
        num_proc_i = num_proc
    if perturb_type == "delete":
        perturbation_dataset = perturbation_dataset.map(delete_index, num_proc=num_proc_i)
    elif perturb_type == "overexpress":
        perturbation_dataset = perturbation_dataset.map(overexpress_index, num_proc=num_proc_i)
    return perturbation_dataset, [perturbation_dataset.data['perturb_index'].to_pylist()]


# original cell emb removing the respective perturbed gene emb
def make_comparison_batch(original_emb, indices_to_perturb):
    all_embs_list = []
    if type(indices_to_perturb) == int:
        indices_to_perturb = [[indices_to_perturb]]
    for indices in indices_to_perturb:
        emb_list = []
        start = 0
        if isinstance(indices[0], list):
        # if len(indices) > 1 and isinstance(indices[0], list):
            indices = flatten_list(indices)
        for i in sorted(indices):
            if type(i) == int:
                emb_list += [original_emb[start:i]]
            else:
                emb_list += [original_emb[start:i[0]]]
            if type(i) == int:
                start = i+1
            else:
                start = i[0]+1
        emb_list += [original_emb[start:]]
        all_embs_list += [torch.cat(emb_list)]
    return torch.stack(all_embs_list)

# average embedding position of goal cell states
def get_cell_state_avg_embs(model,
                            filtered_input_data,
                            cell_states_to_model,
                            layer_to_quant,
                            token_dictionary,
                            forward_batch_size,
                            num_proc):
    possible_states = [value[0] + value[1] + value[2] for value in cell_states_to_model.values()][0]
    state_embs_dict = dict()
    for possible_state in possible_states:
        state_embs_list = []

        total_batch_length = len(filtered_input_data)
        if ((total_batch_length - 1) / forward_batch_size).is_integer():
            forward_batch_size = forward_batch_size - 1

        max_len = max(filtered_input_data["length"])
        for i in tqdm.tqdm(range(0, total_batch_length, forward_batch_size)):
            max_range = min(i + forward_batch_size, total_batch_length)

            state_minibatch = filtered_input_data.select([i for i in range(i, max_range)])
            state_minibatch.set_format(type="torch")

            input_data_minibatch = state_minibatch["input_ids"]
            input_data_minibatch = pad_tensor_list(input_data_minibatch, max_len, token_dictionary)
            with torch.no_grad():
                outputs = model(input_data_minibatch.long().to('cuda:1'))

            state_embs_i = outputs.hidden_states[layer_to_quant]
            state_embs_list += [state_embs_i]
            del outputs
            del state_minibatch
            del input_data_minibatch
            del state_embs_i
            torch.cuda.empty_cache()
        state_embs_stack = torch.cat(state_embs_list)
        state_embs_stack[torch.isnan(state_embs_stack)] = 0#新增
        avg_state_emb = torch.mean(state_embs_stack, dim=[0, 1], keepdim=True)
        state_embs_dict[possible_state] = avg_state_emb
    return state_embs_dict


# quantify cosine similarity of perturbed vs original or alternate states
def quant_cos_sims(model,
                   perturbation_batch,
                   forward_batch_size,
                   layer_to_quant,
                   perturb_type,
                   original_emb,
                   indices_to_perturb,
                   cell_states_to_model,
                   state_embs_dict):
    cos = torch.nn.CosineSimilarity(dim=2)
    original_emb = torch.nan_to_num(original_emb, nan=0.0)
    total_batch_length = len(perturbation_batch)
    if ((total_batch_length - 1) / forward_batch_size).is_integer():
        forward_batch_size = forward_batch_size - 1
    if cell_states_to_model is None:
        comparison_batch = make_comparison_batch(original_emb, indices_to_perturb)
        cos_sims = []
    else:
        possible_states = [value[0] + value[1] + value[2] for value in cell_states_to_model.values()][0]
        cos_sims_vs_alt_dict = dict(zip(possible_states, [[] for i in range(len(possible_states))]))
    for i in tqdm.tqdm(range(0, total_batch_length, forward_batch_size)):
        max_range = min(i + forward_batch_size, total_batch_length)

        perturbation_minibatch = perturbation_batch.select([i for i in range(i, max_range)])
        perturbation_minibatch.set_format(type="torch")

        input_data_minibatch = perturbation_minibatch.data["input_ids"].to_pylist()
        input_data_minibatch = torch.tensor(input_data_minibatch)
        # if input_data_minibatch.shape[1] != 2048:
        #     zeros = torch.zeros((input_data_minibatch.shape[0], 1, input_data_minibatch.shape[-1]))
        #     input_data_minibatch = torch.cat((input_data_minibatch, zeros), dim=1)
        with torch.no_grad():
            outputs = model.bert.forward(input_data_minibatch[:,:,0].long().to('cuda:1'), input_data_minibatch[:,:,1].to('cuda:1'),
                                         species=torch.zeros(input_data_minibatch.size()[0], 1).long().to('cuda:1'))[0]
        del input_data_minibatch
        del perturbation_minibatch
        # cosine similarity between original emb and batch items
            # if len(indices_to_perturb) > 1:
            #     minibatch_emb = torch.squeeze(outputs.hidden_states[layer_to_quant])
            # else:
        minibatch_emb = outputs[:,1:,:]
            # minibatch_emb = minibatch_emb.squeeze(dim=0)
        if cell_states_to_model is None:
            minibatch_comparison = comparison_batch[i:max_range]
            if minibatch_comparison.shape[1]!=minibatch_emb.shape[1]:
                zeros = torch.zeros((minibatch_comparison.shape[0], minibatch_emb.shape[1] - minibatch_comparison.shape[1], minibatch_comparison.shape[-1])).to('cuda:1')
                minibatch_comparison = torch.cat((minibatch_comparison, zeros), dim=1)
            cos_sims += [cos(minibatch_emb, minibatch_comparison).to("cpu")]
        else:
            for state in possible_states:
                cos_sims_vs_alt_dict[state] += cos_sim_shift(original_emb, minibatch_emb, state_embs_dict[state])
        del outputs
        del minibatch_emb
        if cell_states_to_model is None:
            del minibatch_comparison
        torch.cuda.empty_cache()
    if cell_states_to_model is None:
        cos_sims_stack = torch.cat(cos_sims)
        return cos_sims_stack
    else:
        for state in possible_states:
            cos_sims_vs_alt_dict[state] = torch.cat(cos_sims_vs_alt_dict[state])
        return cos_sims_vs_alt_dict


# calculate cos sim shift of perturbation with respect to origin and alternative cell
def cos_sim_shift(original_emb, minibatch_emb, alt_emb):
    cos = torch.nn.CosineSimilarity(dim=2)
    original_emb = torch.mean(original_emb, dim=0, keepdim=True)[None, :]
    alt_emb = alt_emb[None, None, :]
    origin_v_end = cos(original_emb, alt_emb)
    perturb_v_end = cos(torch.mean(minibatch_emb, dim=1, keepdim=True), alt_emb)
    return [(perturb_v_end - origin_v_end).to("cpu")]


# pad list of tensors and convert to tensor
def pad_tensor_list(tensor_list, dynamic_or_constant, token_dictionary):
    pad_token_id = token_dictionary.get("<pad>")
    dynamic_or_constant =dynamic_or_constant[0]
    # Determine maximum tensor length
    if dynamic_or_constant == "dynamic":
        max_len = max([tensor.squeeze().numel() for tensor in tensor_list])
    elif type(dynamic_or_constant) == int:
        max_len = dynamic_or_constant
    else:
        logger.warning(
            "If padding style is constant, must provide integer value. " \
            "Setting padding to max input size 2048.")

    t_list = []
    for tensor in tensor_list:
        t_list.append(torch.tensor(eval(tensor)))
    return torch.stack(t_list)


class InSilicoPerturber:
    valid_option_dict = {
        "perturb_type": {"delete", "overexpress", "inhibit", "activate"},
        "perturb_rank_shift": {None, int},
        "genes_to_perturb": {"all", list},
        "combos": {0, 1, 2},
        "anchor_gene": {None, str},
        "num_classes": {int},
        "emb_mode": {"cell", "cell_and_gene"},
        "cell_emb_style": {"mean_pool"},
        "filter_data": {None, dict},
        "cell_states_to_model": {None, dict},
        "max_ncells": {None, int},
        "emb_layer": {-1, 0},
        "forward_batch_size": {int},
        "nproc": {int},
        "save_raw_data": {False, True},
    }

    def __init__(
            self,
            perturb_type="delete",
            perturb_rank_shift=None,
            genes_to_perturb="all",
            combos=0,
            anchor_gene=None,
            num_classes=0,
            emb_mode="cell",
            cell_emb_style="mean_pool",
            filter_data=None,
            cell_states_to_model=None,
            max_ncells=None,
            emb_layer=-1,
            forward_batch_size=100,
            nproc=4,
            save_raw_data=False,
            token_dictionary_file=None,
    ):
        """
        Initialize in silico perturber.
        Parameters
        ----------
        perturb_type : {"delete","overexpress","inhibit","activate"}
            Type of perturbation.
            "delete": delete gene from rank value encoding
            "overexpress": move gene to front of rank value encoding
            "inhibit": move gene to lower quartile of rank value encoding
            "activate": move gene to higher quartile of rank value encoding
        perturb_rank_shift : None, int
            Number of quartiles by which to shift rank of gene.
            For example, if perturb_type="activate" and perturb_rank_shift=1:
                genes in 4th quartile will move to middle of 3rd quartile.
                genes in 3rd quartile will move to middle of 2nd quartile.
                genes in 2nd quartile will move to middle of 1st quartile.
                genes in 1st quartile will move to front of rank value encoding.
            For example, if perturb_type="inhibit" and perturb_rank_shift=2:
                genes in 1st quartile will move to middle of 3rd quartile.
                genes in 2nd quartile will move to middle of 4th quartile.
                genes in 3rd or 4th quartile will move to bottom of rank value encoding.
        genes_to_perturb : "all", list
            Default is perturbing each gene detected in each cell in the dataset.
            Otherwise, may provide a list of ENSEMBL IDs of genes to perturb.
        combos : {0,1,2}
            Whether to perturb genes individually (0), in pairs (1), or in triplets (2).
        anchor_gene : None, str
            ENSEMBL ID of gene to use as anchor in combination perturbations.
            For example, if combos=1 and anchor_gene="ENSG00000148400":
                anchor gene will be perturbed in combination with each other gene.
        num_classes : int
            If model is a gene or cell classifier, specify number of classes it was trained to classify.
            For the pretrained Geneformer model, number of classes is 0 as it is not a classifier.
        emb_mode : {"cell","cell_and_gene"}
            Whether to output impact of perturbation on cell and/or gene embeddings.
        cell_emb_style : "mean_pool"
            Method for summarizing cell embeddings.
            Currently only option is mean pooling of gene embeddings for given cell.
        filter_data : None, dict
            Default is to use all input data for in silico perturbation study.
            Otherwise, dictionary specifying .dataset column name and list of values to filter by.
        cell_states_to_model: None, dict
            Cell states to model if testing perturbations that achieve goal state change.
            Single-item dictionary with key being cell attribute (e.g. "disease").
            Value is tuple of three lists indicating start state, goal end state, and alternate possible end states.
        max_ncells : None, int
            Maximum number of cells to test.
            If None, will test all cells.
        emb_layer : {-1, 0}
            Embedding layer to use for quantification.
            -1: 2nd to last layer (recommended for pretrained Geneformer)
            0: last layer (recommended for cell classifier fine-tuned for disease state)
        forward_batch_size : int
            Batch size for forward pass.
        nproc : int
            Number of CPU processes to use.
        save_raw_data: {False,True}
            Whether to save raw perturbation data for each gene/cell.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl ID:token).
        """

        self.perturb_type = perturb_type
        self.perturb_rank_shift = perturb_rank_shift
        self.genes_to_perturb = genes_to_perturb
        self.combos = combos
        self.anchor_gene = anchor_gene
        self.num_classes = num_classes
        self.emb_mode = emb_mode
        self.cell_emb_style = cell_emb_style
        self.filter_data = filter_data
        self.cell_states_to_model = cell_states_to_model
        self.max_ncells = max_ncells
        self.emb_layer = emb_layer
        self.forward_batch_size = forward_batch_size
        self.nproc = nproc
        self.save_raw_data = save_raw_data

        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        if anchor_gene is None:
            self.anchor_token = None
        else:
            self.anchor_token = self.gene_token_dict[self.anchor_gene]

        if genes_to_perturb == "all":
            self.tokens_to_perturb = "all"
        else:
            self.tokens_to_perturb = self.gene_token_dict[genes_to_perturb]

    def validate_options(self):
        for attr_name, valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int, list, dict]) and isinstance(attr_value, option):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. " \
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise

        if self.perturb_type in ["delete", "overexpress"]:
            if self.perturb_rank_shift is not None:
                if self.perturb_type == "delete":
                    logger.warning(
                        "perturb_rank_shift set to None. " \
                        "If perturb type is delete then gene is deleted entirely " \
                        "rather than shifted by quartile")
                elif self.perturb_type == "overexpress":
                    logger.warning(
                        "perturb_rank_shift set to None. " \
                        "If perturb type is activate then gene is moved to front " \
                        "of rank value encoding rather than shifted by quartile")
            self.perturb_rank_shift = None

        if (self.anchor_gene is not None) and (self.emb_mode == "cell_and_gene"):
            self.emb_mode = "cell"
            logger.warning(
                "emb_mode set to 'cell'. " \
                "Currently, analysis with anchor gene " \
                "only outputs effect on cell embeddings.")

        if self.cell_states_to_model is not None:
            if (len(self.cell_states_to_model.items()) == 1):
                for key, value in self.cell_states_to_model.items():
                    if (len(value) == 3) and isinstance(value, tuple):
                        if isinstance(value[0], list) and isinstance(value[1], list) and isinstance(value[2], list):
                            if len(value[0]) == 1 and len(value[1]) == 1:
                                all_values = value[0] + value[1] + value[2]
                                if len(all_values) == len(set(all_values)):
                                    continue
            else:
                logger.error(
                    "Cell states to model must be a single-item dictionary with " \
                    "key being cell attribute (e.g. 'disease') and value being " \
                    "tuple of three lists indicating start state, goal end state, and alternate possible end states. " \
                    "Values should all be unique. " \
                    "For example: {'disease':(['dcm'],['ctrl'],['hcm'])}")
                raise
            if self.anchor_gene is not None:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene set to None. " \
                    "Currently, anchor gene not available " \
                    "when modeling multiple cell states.")

        if self.perturb_type in ["inhibit", "activate"]:
            if self.perturb_rank_shift is None:
                logger.error(
                    "If perturb type is inhibit or activate then " \
                    "quartile to shift by must be specified.")
                raise

        if self.filter_data is not None:
            for key, value in self.filter_data.items():
                if type(value) != list:
                    self.filter_data[key] = [value]
                    logger.warning(
                        "Values in filter_data dict must be lists. " \
                        f"Changing {key} value to list ([{value}]).")

    def perturb_data(self,
                     model_directory,
                     input_data_file,
                     output_directory,
                     output_prefix):
        """
        Perturb genes in input data and save as results in output_directory.
        Parameters
        ----------
        model_directory : Path
            Path to directory containing model
        input_data_file : Path
            Path to directory containing .dataset inputs
        output_directory : Path
            Path to directory where perturbation data will be saved as .pickle
        output_prefix : str
            Prefix for output .dataset
        """

        filtered_input_data = self.load_and_filter(input_data_file)
        df = filtered_input_data.to_pandas()
        df['species'] = 0 # 0:human; 1:mouse
        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int32')),
            'values': Sequence(feature=Value(dtype='float32')),
            'length': Sequence(feature=Value(dtype='int16')),
            'cell_type': Value(dtype='string'),
            'species': Value(dtype='int32')
        })
        filtered_input_data = Dataset.from_pandas(df, features=features)
        model = self.load_model(model_directory)
        layer_to_quant = quant_layers(model) + self.emb_layer

        state_embs_dict = None
        self.in_silico_perturb(model,
                               filtered_input_data,
                               layer_to_quant,
                               state_embs_dict,
                               output_directory,
                               output_prefix)

    # load data and filter by defined criteria
    def load_and_filter(self, input_data_file):
        data = load_from_disk(input_data_file)
        if self.filter_data is not None:
            for key, value in self.filter_data.items():
                def filter_data_by_criteria(example):
                    return example[key] in value

                data = data.filter(filter_data_by_criteria, num_proc=self.nproc)
            if len(data) == 0:
                logger.error(
                    "No cells remain after filtering. Check filtering criteria.")
                raise
        data_shuffled = data.shuffle(seed=42)
        num_cells = len(data_shuffled)
        # if max number of cells is defined, then subsample to this max number
        if self.max_ncells != None:
            num_cells = min(self.max_ncells, num_cells)
        data_subset = data_shuffled.select([i for i in range(num_cells)])
        # sort dataset with largest cell first to encounter any memory errors earlier
        # data_sorted = data_subset.sort("length", reverse=True)
        return data_subset

    # load model to GPU
    def load_model(self, model_directory):
        model = BertForMaskedLM.from_pretrained(model_directory,
                                                knowledges=knowledges,
                                                output_hidden_states=True,
                                                output_attentions=False)
        model.eval()
        model = model.to("cuda:1")
        return model

    # determine effect of perturbation on other genes
    def in_silico_perturb(self,
                          model,
                          filtered_input_data,
                          layer_to_quant,
                          state_embs_dict,
                          output_directory,
                          output_prefix):

        output_path_prefix = f"{output_directory}{output_prefix}"
        # filter dataset for cells that have tokens to be perturbed
        if self.anchor_token is not None:
            def if_has_tokens_to_perturb(example):
                return (len(set(example["input_ids"]).intersection(self.anchor_token)) == len(self.anchor_token))

            filtered_input_data = filtered_input_data.filter(if_has_tokens_to_perturb, num_proc=self.nproc)
            logger.info(f"# cells with anchor gene: {len(filtered_input_data)}")
        if self.tokens_to_perturb != "all":
            def if_has_tokens_to_perturb(example):
                return (len(set(example["input_ids"]).intersection(self.tokens_to_perturb)) > 0)

            filtered_input_data = filtered_input_data.filter(if_has_tokens_to_perturb, num_proc=self.nproc)
        cos_sims_dict = defaultdict(list)
        pickle_batch = -1
        for i in tqdm.tqdm(trange(len(filtered_input_data))):
            example_cell = filtered_input_data.select([i])
            original_emb = forward_pass_single_cell(model, example_cell, layer_to_quant)
            input_ids = example_cell["input_ids"][0].unsqueeze(1)
            values = example_cell["values"][0].unsqueeze(1)
            input = torch.cat((input_ids, values), dim=1)
            gene_list = torch.squeeze(input)

            # reset to original type to prevent downstream issues due to forward_pass_single_cell modifying as torch format in place
            example_cell = filtered_input_data.select([i])

            if self.anchor_token is None:
                for combo_lvl in range(self.combos + 1):
                    perturbation_batch, indices_to_perturb = make_perturbation_batch(example_cell,
                                                                                     self.perturb_type,
                                                                                     self.tokens_to_perturb,
                                                                                     self.anchor_token,
                                                                                     combo_lvl,
                                                                                     self.nproc)
                    cos_sims_data = quant_cos_sims(model,
                                                   perturbation_batch,
                                                   self.forward_batch_size,
                                                   layer_to_quant,
                                                   self.perturb_type,
                                                   original_emb,
                                                   indices_to_perturb,
                                                   self.cell_states_to_model,
                                                   state_embs_dict)

                    for j in range(cos_sims_data.shape[0]):
                        if self.genes_to_perturb != "all":
                            j_index = torch.tensor(indices_to_perturb[0][j])
                            if j_index.shape[0] > 1:
                                j_index = torch.squeeze(j_index)
                        else:
                            j_index = torch.tensor([j])
                        if j_index.dim() == 2:
                            j_index =j_index[0]
                        perturbed_gene = torch.index_select(gene_list, 0, j_index)

                        if perturbed_gene[0].shape[0] == 1:
                            perturbed_gene = perturbed_gene.item()
                        elif perturbed_gene[0].shape[0] > 1:
                            perturbed_gene = tuple(perturbed_gene.tolist())

                        cell_cos_sim = torch.mean(cos_sims_data[j]).item()
                        cos_sims_dict[(int(perturbed_gene[0][0]), "cell_emb")] += [cell_cos_sim]

                        if self.emb_mode == "cell_and_gene":
                            for k in range(cos_sims_data.shape[1]):
                                cos_sim_value = cos_sims_data[j][k]
                                affected_gene = int(gene_list[k][0].item())
                                cos_sims_dict[(int(perturbed_gene[0][0]), affected_gene)] += [cos_sim_value.item()]

            elif self.anchor_token is not None:
                perturbation_batch, indices_to_perturb = make_perturbation_batch(example_cell,
                                                                                 self.perturb_type,
                                                                                 self.tokens_to_perturb,
                                                                                 None,
                                                                                 0,
                                                                                 self.nproc)
                cos_sims_data = quant_cos_sims(model,
                                               perturbation_batch,
                                               self.forward_batch_size,
                                               layer_to_quant,
                                               self.perturb_type,
                                               original_emb,
                                               indices_to_perturb,
                                               self.cell_states_to_model,
                                               state_embs_dict)
                cos_sims_data = cos_sims_data.to("cuda:1")

                combo_perturbation_batch, combo_indices_to_perturb = make_perturbation_batch(example_cell,
                                                                                             self.perturb_type,
                                                                                             self.tokens_to_perturb,
                                                                                             self.anchor_token,
                                                                                             1,
                                                                                             self.nproc)
                combo_cos_sims_data = quant_cos_sims(model,
                                                     combo_perturbation_batch,
                                                     self.forward_batch_size,
                                                     layer_to_quant,
                                                     original_emb,
                                                     combo_indices_to_perturb,
                                                     self.cell_states_to_model,
                                                     state_embs_dict)
                combo_cos_sims_data = combo_cos_sims_data.to("cuda:1")

                # update cos sims dict
                # key is tuple of (perturbed_gene, "cell_emb") for avg cell emb change
                anchor_index = example_cell["input_ids"][0].index(self.anchor_token[0])
                anchor_cell_cos_sim = torch.mean(cos_sims_data[anchor_index]).item()
                non_anchor_indices = [k for k in range(cos_sims_data.shape[0]) if k != anchor_index]
                cos_sims_data = cos_sims_data[non_anchor_indices, :]

                for j in range(cos_sims_data.shape[0]):

                    if j < anchor_index:
                        j_index = torch.tensor([j])
                    else:
                        j_index = torch.tensor([j + 1])

                    perturbed_gene = torch.index_select(gene_list, 0, j_index)
                    perturbed_gene = perturbed_gene.item()

                    cell_cos_sim = torch.mean(cos_sims_data[j]).item()
                    combo_cos_sim = torch.mean(combo_cos_sims_data[j]).item()
                    cos_sims_dict[(perturbed_gene, "cell_emb")] += [(anchor_cell_cos_sim,  # cos sim anchor gene alone
                                                                     cell_cos_sim,  # cos sim deleted gene alone
                                                                     combo_cos_sim)]  # cos sim anchor gene + deleted gene

        with open(f"{output_path_prefix}_{pickle_batch}.pickle", "wb") as fp:
            pickle.dump(cos_sims_dict, fp)

if __name__ == '__main__':

    isp = InSilicoPerturber(perturb_type="delete",
                            perturb_rank_shift=None,
                            genes_to_perturb=["ENSID"],
                            combos=0,
                            anchor_gene=None,
                            num_classes=0,
                            emb_mode="cell_and_gene",
                            cell_emb_style="mean_pool",
                            filter_data=None,
                            cell_states_to_model=None,
                            max_ncells=None,
                            emb_layer=-1,
                            forward_batch_size=10,
                            nproc=32,
                            save_raw_data=False)

    isp.perturb_data("/home/ict/compass_r2/outputs/models/240613_055744_mta2_values_promoter_co-exp_gene-family_peca-grn_L12_emb768_SL2048_E3_B4_LR0.001_LSlinear_WU1000_Oadamw_DS5/checkpoint-100",   #model
                     "/home/ict/genecompass_data/mouse_9/scRNA_GSM4476786_homo",  #input
                     "/home/ict/genecompass_data/mouse_9/mta2_ft/",    #output
                     "Delete")    #
    print()
    """
        isp.perturb_data("path/to/model",
                         "path/to/input_data",
                         "path/to/output_directory",
                         "output_prefix")
    """