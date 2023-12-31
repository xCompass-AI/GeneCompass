"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

import pickle
from pathlib import Path

import loompy as lp
import numpy as np
from datasets import Dataset

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector[nonzero_mask])
    # tokenize
    sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
    return sentence_tokens


class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict,
        nproc=1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))

    def tokenize_data(self, loom_data_directory, output_directory, output_prefix):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        """
        tokenized_cells, cell_metadata = self.tokenize_files(Path(loom_data_directory))
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata)

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)

    def tokenize_files(self, loom_data_directory):
        tokenized_cells = []
        cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.keys()}

        # loops through directories to tokenize .loom files
        for loom_file_path in loom_data_directory.glob("*.loom"):
            print(f"Tokenizing {loom_file_path}")
            file_tokenized_cells, file_cell_metadata = self.tokenize_file(
                loom_file_path
            )
            tokenized_cells += file_tokenized_cells
            for k in cell_metadata.keys():
                cell_metadata[k] += file_cell_metadata[k]

        return tokenized_cells, cell_metadata

    def tokenize_file(self, loom_file_path):
        file_cell_metadata = {
            attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
        }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists is True:
                filter_pass_loc = np.where(
                    [True if i == 1 else False for i in data.ca["filter_pass"]]
                )[0]
            elif var_exists is False:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / subview.ca.n_counts
                    * 10_000
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += subview.ca[k].tolist()

        return tokenized_cells, file_cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata):
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        dataset_dict.update(cell_metadata)

        # create dataset
        output_dataset = Dataset.from_dict(dataset_dict)

        # truncate dataset
        def truncate(example):
            example["input_ids"] = example["input_ids"][0:2048]
            return example

        output_dataset_truncated = output_dataset.map(truncate, num_proc=self.nproc)

        # measure lengths of dataset
        def measure_length(example):
            example["length"] = len(example["input_ids"])
            return example

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, num_proc=self.nproc
        )

        return output_dataset_truncated_w_length
