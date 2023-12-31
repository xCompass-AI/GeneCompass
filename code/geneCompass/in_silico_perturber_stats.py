"""
Geneformer in silico perturber stats generator.

Usage:
  from geneformer import InSilicoPerturberStats
  ispstats = InSilicoPerturberStats(mode="goal_state_shift",
                                    combos=0,
                                    anchor_gene=None,
                                    cell_states_to_model={"disease":(["dcm"],["ctrl"],["hcm"])})
  ispstats.get_stats("path/to/input_data",
                     None,
                     "path/to/output_directory",
                     "output_prefix")
"""


import os
import logging
import numpy as np
import pandas as pd
import pickle
import random
import statsmodels.stats.multitest as smt
from pathlib import Path
from scipy.stats import ranksums
from tqdm.notebook import trange

from .tokenizer import TOKEN_DICTIONARY_FILE

GENE_NAME_ID_DICTIONARY_FILE = Path(__file__).parent / "gene_name_id_dict.pkl"

logger = logging.getLogger(__name__)

# invert dictionary keys/values
def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}

# read raw dictionary files
def read_dictionaries(dir, cell_or_gene_emb):
    dict_list = []
    for file in os.listdir(dir):
        # process only _raw.pickle files
        if file.endswith("_raw.pickle"):
            with open(f"{dir}/{file}", "rb") as fp:
                cos_sims_dict = pickle.load(fp)
                if cell_or_gene_emb == "cell":
                    cell_emb_dict = {k: v for k,
                                    v in cos_sims_dict.items() if v and "cell_emb" in k}
                dict_list += [cell_emb_dict]
    return dict_list

# get complete gene list
def get_gene_list(dict_list):
    gene_set = set()
    for dict_i in dict_list:
        gene_set.update([k[0] for k, v in dict_i.items() if v])
    gene_list = list(gene_set)
    gene_list.sort()
    return gene_list

def n_detections(token, dict_list):
    cos_sim_megalist = []
    for dict_i in dict_list:
        cos_sim_megalist += dict_i.get((token, "cell_emb"),[])
    return len(cos_sim_megalist)

def get_fdr(pvalues):
    return list(smt.multipletests(pvalues, alpha=0.05, method="fdr_bh")[1])

# stats comparing cos sim shifts towards goal state of test perturbations vs random perturbations
def isp_stats_to_goal_state(cos_sims_df, dict_list):
    random_tuples = []
    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        for dict_i in dict_list:
            random_tuples += dict_i.get((token, "cell_emb"),[])
    goal_end_random_megalist = [goal_end for goal_end,alt_end,start_state in random_tuples]
    alt_end_random_megalist = [alt_end for goal_end,alt_end,start_state in random_tuples]
    start_state_random_megalist = [start_state for goal_end,alt_end,start_state in random_tuples]
    
    # downsample to improve speed of ranksums
    if len(goal_end_random_megalist) > 100_000:
        random.seed(42)
        goal_end_random_megalist = random.sample(goal_end_random_megalist, k=100_000)
    if len(alt_end_random_megalist) > 100_000:
        random.seed(42)
        alt_end_random_megalist = random.sample(alt_end_random_megalist, k=100_000)
    if len(start_state_random_megalist) > 100_000:
        random.seed(42)
        start_state_random_megalist = random.sample(start_state_random_megalist, k=100_000)
    
    names=["Gene",
           "Gene_name",
           "Ensembl_ID",
           "Shift_from_goal_end",
           "Shift_from_alt_end",
           "Goal_end_vs_random_pval",
           "Alt_end_vs_random_pval"]
    cos_sims_full_df = pd.DataFrame(columns=names)
    
    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        name = cos_sims_df["Gene_name"][i]
        ensembl_id = cos_sims_df["Ensembl_ID"][i]
        token_tuples = []
        
        for dict_i in dict_list:
            token_tuples += dict_i.get((token, "cell_emb"),[])
        
        goal_end_cos_sim_megalist = [goal_end for goal_end,alt_end,start_state in token_tuples]
        alt_end_cos_sim_megalist = [alt_end for goal_end,alt_end,start_state in token_tuples]
        
        mean_goal_end = np.mean(goal_end_cos_sim_megalist)
        mean_alt_end = np.mean(alt_end_cos_sim_megalist)
        
        pval_goal_end = ranksums(goal_end_random_megalist,goal_end_cos_sim_megalist).pvalue
        pval_alt_end = ranksums(alt_end_random_megalist,alt_end_cos_sim_megalist).pvalue
        
        data_i = [token, 
                  name,
                  ensembl_id,
                  mean_goal_end, 
                  mean_alt_end,
                  pval_goal_end,
                  pval_alt_end]
        
        cos_sims_df_i = pd.DataFrame(dict(zip(names,data_i)),index=[i])
        cos_sims_full_df = pd.concat([cos_sims_full_df,cos_sims_df_i])
        
    cos_sims_full_df["Goal_end_FDR"] = get_fdr(list(cos_sims_full_df["Goal_end_vs_random_pval"]))
    cos_sims_full_df["Alt_end_FDR"] = get_fdr(list(cos_sims_full_df["Alt_end_vs_random_pval"]))
    
    return cos_sims_full_df

# stats comparing cos sim shifts of test perturbations vs null distribution
def isp_stats_vs_null(cos_sims_df, dict_list, null_dict_list):
    cos_sims_full_df = cos_sims_df.copy()

    cos_sims_full_df["Test_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Null_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_v_null_avg_shift"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_v_null_pval"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["Test_v_null_FDR"] = np.zeros(cos_sims_df.shape[0], dtype=float)
    cos_sims_full_df["N_Detections_test"] = np.zeros(cos_sims_df.shape[0], dtype="uint32")
    cos_sims_full_df["N_Detections_null"] = np.zeros(cos_sims_df.shape[0], dtype="uint32")
    
    for i in trange(cos_sims_df.shape[0]):
        token = cos_sims_df["Gene"][i]
        test_shifts = []
        null_shifts = []
        
        for dict_i in dict_list:
            test_shifts += dict_i.get((token, "cell_emb"),[])

        for dict_i in null_dict_list:
            null_shifts += dict_i.get((token, "cell_emb"),[])
        
        cos_sims_full_df.loc[i, "Test_avg_shift"] = np.mean(test_shifts)
        cos_sims_full_df.loc[i, "Null_avg_shift"] = np.mean(null_shifts)
        cos_sims_full_df.loc[i, "Test_v_null_avg_shift"] = np.mean(test_shifts)-np.mean(null_shifts)       
        cos_sims_full_df.loc[i, "Test_v_null_pval"] = ranksums(test_shifts,
            null_shifts, nan_policy="omit").pvalue

        cos_sims_full_df.loc[i, "N_Detections_test"] = len(test_shifts)
        cos_sims_full_df.loc[i, "N_Detections_null"] = len(null_shifts)

    cos_sims_full_df["Test_v_null_FDR"] = get_fdr(cos_sims_full_df["Test_v_null_pval"])
    return cos_sims_full_df

class InSilicoPerturberStats:
    valid_option_dict = {
        "mode": {"goal_state_shift","vs_null","vs_random"},
        "combos": {0,1,2},
        "anchor_gene": {None, str},
        "cell_states_to_model": {None, dict},
    }
    def __init__(
        self,
        mode="vs_random",
        combos=0,
        anchor_gene=None,
        cell_states_to_model=None,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        gene_name_id_dictionary_file=GENE_NAME_ID_DICTIONARY_FILE,
    ):
        """
        Initialize in silico perturber stats generator.

        Parameters
        ----------
        mode : {"goal_state_shift","vs_null","vs_random"}
            Type of stats.
            "goal_state_shift": perturbation vs. random for desired cell state shift
            "vs_null": perturbation vs. null from provided null distribution dataset
            "vs_random": perturbation vs. random gene perturbations in that cell (no goal direction)
        combos : {0,1,2}
            Whether to perturb genes individually (0), in pairs (1), or in triplets (2).
        anchor_gene : None, str
            ENSEMBL ID of gene to use as anchor in combination perturbations.
            For example, if combos=1 and anchor_gene="ENSG00000148400":
                anchor gene will be perturbed in combination with each other gene.
        cell_states_to_model: None, dict
            Cell states to model if testing perturbations that achieve goal state change.
            Single-item dictionary with key being cell attribute (e.g. "disease").
            Value is tuple of three lists indicating start state, goal end state, and alternate possible end states.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl ID:token).
        gene_name_id_dictionary_file : Path
            Path to pickle file containing gene name to ID dictionary (gene name:Ensembl ID).
        """

        self.mode = mode
        self.combos = combos
        self.anchor_gene = anchor_gene
        self.cell_states_to_model = cell_states_to_model
        
        self.validate_options()

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)
            
        # load gene name dictionary (gene name:Ensembl ID)
        with open(gene_name_id_dictionary_file, "rb") as f:
            self.gene_name_id_dict = pickle.load(f)

        if anchor_gene is None:
            self.anchor_token = None
        else:
            self.anchor_token = self.gene_token_dict[self.anchor_gene]

    def validate_options(self):
        for attr_name,valid_options in self.valid_option_dict.items():
            attr_value = self.__dict__[attr_name]
            if type(attr_value) not in {list, dict}:
                if attr_value in valid_options:
                    continue
            valid_type = False
            for option in valid_options:
                if (option in [int,list,dict]) and isinstance(attr_value, option):
                    valid_type = True
                    break
            if valid_type:
                continue
            logger.error(
                f"Invalid option for {attr_name}. " \
                f"Valid options for {attr_name}: {valid_options}"
            )
            raise
        
        if self.cell_states_to_model is not None:
            if (len(self.cell_states_to_model.items()) == 1):
                for key,value in self.cell_states_to_model.items():
                    if (len(value) == 3) and isinstance(value, tuple):
                        if isinstance(value[0],list) and isinstance(value[1],list) and isinstance(value[2],list):
                            if len(value[0]) == 1 and len(value[1]) == 1:
                                all_values = value[0]+value[1]+value[2]
                                if len(all_values) == len(set(all_values)):
                                    continue
            else:
                logger.error(
                    "Cell states to model must be a single-item dictionary with " \
                    "key being cell attribute (e.g. 'disease') and value being " \
                    "tuple of three lists indicating start state, goal end state, and alternate possible end states. " \
                    "Values should all be unique. " \
                    "For example: {'disease':(['start_state'],['ctrl'],['alt_end'])}")
                raise
            if self.anchor_gene is not None:
                self.anchor_gene = None
                logger.warning(
                    "anchor_gene set to None. " \
                    "Currently, anchor gene not available " \
                    "when modeling multiple cell states.")

    def get_stats(self,
                  input_data_directory,
                  null_dist_data_directory,
                  output_directory,
                  output_prefix):
        """
        Get stats for in silico perturbation data and save as results in output_directory.

        Parameters
        ----------
        input_data_directory : Path
            Path to directory containing cos_sim dictionary inputs
        null_dist_data_directory : Path
            Path to directory containing null distribution cos_sim dictionary inputs
        output_directory : Path
            Path to directory where perturbation data will be saved as .csv
        output_prefix : str
            Prefix for output .dataset
        """

        if self.mode not in ["goal_state_shift", "vs_null"]:
            logger.error(
                "Currently, only modes available are stats for goal_state_shift \
                    and vs_null (comparing to null distribution).")
            raise

        self.gene_token_id_dict = invert_dict(self.gene_token_dict)
        self.gene_id_name_dict = invert_dict(self.gene_name_id_dict)

        # obtain total gene list
        dict_list = read_dictionaries(input_data_directory, "cell")
        gene_list = get_gene_list(dict_list)
        
        # initiate results dataframe
        cos_sims_df_initial = pd.DataFrame({"Gene": gene_list, 
                                            "Gene_name": [self.token_to_gene_name(item) \
                                                          for item in gene_list], \
                                            "Ensembl_ID": [self.gene_token_id_dict[genes[1]] \
                                                           if isinstance(genes,tuple) else \
                                                           self.gene_token_id_dict[genes] \
                                                           for genes in gene_list]}, \
                                             index=[i for i in range(len(gene_list))])

        if self.mode == "goal_state_shift":
            cos_sims_df = isp_stats_to_goal_state(cos_sims_df_initial, dict_list)
            
            # quantify number of detections of each gene
            cos_sims_df["N_Detections"] = [n_detections(i, dict_list) for i in cos_sims_df["Gene"]]
            
            # sort by shift to desired state
            cos_sims_df = cos_sims_df.sort_values(by=["Shift_from_goal_end",
                                                      "Goal_end_FDR"])
        elif self.mode == "vs_null":
            dict_list = read_dictionaries(input_data_directory, "cell")
            null_dict_list = read_dictionaries(null_dist_data_directory, "cell")
            cos_sims_df = isp_stats_vs_null(cos_sims_df_initial, dict_list,
                null_dict_list)
            cos_sims_df = cos_sims_df.sort_values(by=["Test_v_null_avg_shift",
                                                      "Test_v_null_FDR"])

        # save perturbation stats to output_path
        output_path = (Path(output_directory) / output_prefix).with_suffix(".csv")
        cos_sims_df.to_csv(output_path)

    def token_to_gene_name(self, item):
        if isinstance(item,int):
            return self.gene_id_name_dict.get(self.gene_token_id_dict.get(item, np.nan), np.nan)
        if isinstance(item,tuple):
            return tuple([self.gene_id_name_dict.get(self.gene_token_id_dict.get(i, np.nan), np.nan) for i in item])
