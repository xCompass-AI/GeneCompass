#!/usr/bin/env python
# coding: utf-8

# run with:
# deepspeed --include localhost:0,1,2,3,4,5,6,7 pretrain_modified_geneformer_w_human_mouse_120W.py --deepspeed ds_config.json
# WANDB_MODE=disabled deepspeed --include localhost:0,1,2,3,4,5,6,7 pretrain_modified_geneformer_w_human_mouse_120W.py --deepspeed ds_config.json

import datetime

# imports
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"

import pickle
import random
import subprocess

import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, TrainingArguments

from geneformer import GeneformerPretrainer, BertForMaskedLM

seed_num = 0
random.seed(seed_num)
np.random.seed(seed_num)
seed_val = 42
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# set local time/directories
timezone = pytz.timezone("US/Eastern")
rootdir = "./outputs"

# wandb_name = 'test'
wandb_name = '[pretrain]BERT-base_new_concat_HM_100M_epoch1_bs10_8_4_lr5e-5_cls_value_4priors_id02value08'
# paths here
token_dict_path = '/data/5_folder/6000W_pretrain_data_merge/human_mouse_tokens.pickle'  # 6000w
dataset_path = '/data/5_folder/100M_pretrain_data_merge_new/human_mouse_data/'


example_lengths_file = dataset_path + '/sorted_length.pickle'

# prior knowledge
# use_promoter = False
use_promoter = True
# use_co_exp = False
use_co_exp = True
# use_gene_family = False
use_gene_family = True
# use_peca_grn = False
use_peca_grn = True

# important parameters
# use_values = False
use_values = True
# use_value_embed
use_value_embed = False
# concat values first
concat_values_first = False
# add cls
use_cls_token = True

# set model parameters
# model type
model_type = "bert"
# max input size
max_input_size = 2**11  # 2048
# number of layers
num_layers = 12
# number of attention heads
num_attn_heads = 12
# number of embedding dimensions
num_embed_dim = 768
# intermediate size
intermed_size = num_embed_dim * 4
# activation function
activ_fn = "gelu"
# initializer range, layer norm, dropout
initializer_range = 0.02
layer_norm_eps = 1e-12
attention_probs_dropout_prob = 0.02
hidden_dropout_prob = 0.02


# set training parameters
# total number of examples in Genecorpus-30M after QC filtering:
# num_examples = 27_406_208
# number gpus
num_gpus = 8
# batch size for training and eval
geneformer_batch_size = 10
# max learning rate
max_lr = 5e-5
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 10_000
# max steps
max_steps = 172_000 
# number of epochs
epochs = 1
# optimizer
optimizer = "adamw"
# weight_decay
weight_decay = 0.01


# output directories
current_date = datetime.datetime.now(tz=timezone)
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
run_name = f"Base_{datestamp}_{wandb_name}_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
training_output_dir = f"{rootdir}/models/{run_name}/"
logging_dir = f"{rootdir}/runs/{run_name}/"
model_output_dir = os.path.join(training_output_dir, "models/")


# ensure not overwriting previously saved model
model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
if os.path.isfile(model_output_file) is True:
    raise Exception("Model already saved to this directory.")


# make training and model output directories
subprocess.call(f"mkdir {training_output_dir}", shell=True)
subprocess.call(f"mkdir {model_output_dir}", shell=True)


# load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/datasets/token_dictionary.pkl)
# token_dict_path = '/data/home/ia00/wzc/Geneformer/geneformer/token_dictionary.pkl'
with open(token_dict_path, "rb") as fp:
    token_dictionary = pickle.load(fp)


knowledges = dict()
from utils import load_prior_embedding
out = load_prior_embedding(token_dictionary_or_path=token_dict_path)

knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]


# model configuration
config = {
    # "add_cls": False,    # add additional <CLS> token at the start of the sequence
    # "fast_mlm": False,   # only compute masked token while passing through the classification header
    # "value_concat": True,   # directly concat value with id-embedding as the final embedding
    "hidden_size": num_embed_dim,
    "num_hidden_layers": num_layers,
    "initializer_range": initializer_range,
    "layer_norm_eps": layer_norm_eps,
    "attention_probs_dropout_prob": attention_probs_dropout_prob,
    "hidden_dropout_prob": hidden_dropout_prob,
    "intermediate_size": intermed_size,
    "hidden_act": activ_fn,
    "max_position_embeddings": max_input_size,
    "model_type": model_type,
    "num_attention_heads": num_attn_heads,
    "pad_token_id": token_dictionary.get("<pad>"),
    "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
    "use_values": use_values,
    "use_promoter": use_promoter,
    "use_co_exp": use_co_exp,
    "use_gene_family": use_gene_family,
    "use_peca_grn": use_peca_grn,
    "warmup_steps": warmup_steps,
    "concat_values_first": concat_values_first,
    "use_value_embed": use_value_embed,
    "use_cls_token": use_cls_token,
}

config = BertConfig(**config)
model = BertForMaskedLM(config, knowledges=knowledges)
model = model.train()

# define the training arguments
training_args = {
    "run_name": wandb_name,
    "fp16": True,     # auto mixed precision
    "fp16_opt_level": "O1",   # fp16 level
    "ddp_find_unused_parameters": False,
    # "sharded_ddp": "zero_dp_2",
    # "gradient_checkpointing": True,   # lower gpu memory cost
    # "report_to": 'none',  # online/offline wandb report
    "dataloader_num_workers": 8,
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": False,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "per_device_train_batch_size": geneformer_batch_size,
    # "max_steps": max_steps,
    "num_train_epochs": epochs,
    "save_strategy": "steps",
    # "save_steps": np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
    "save_steps": 10000,  # 
    "logging_steps": 100,
    "output_dir": training_output_dir,
    "logging_dir": logging_dir,
}
training_args = TrainingArguments(**training_args)

print("Starting training.")

# define the trainer
trainer = GeneformerPretrainer(
    model=model,
    args=training_args,
    # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
    train_dataset=load_from_disk(dataset_path),
    # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048_sorted_lengths.pkl)
    example_lengths_file=example_lengths_file,
    token_dictionary=token_dictionary,
)

# train
trainer.train()

# save model
trainer.save_model(model_output_dir)
