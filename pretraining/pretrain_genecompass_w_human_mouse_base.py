#!/usr/bin/env python
# coding: utf-8

import os
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["OMPI_MCA_opal_cuda_support"] = "true"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
os.environ["WANDB_MODE"]="offline"
import pickle
import random
import datetime
import subprocess
import numpy as np
import pytz
import argparse

import torch
import torch.distributed as dist

from transformers import BertConfig, TrainingArguments
from datasets import load_from_disk, disable_caching
disable_caching()
from genecompass import GenecompassPretrainer, BertForMaskedLM
from genecompass.utils import load_prior_embedding


def main(args):
    # Set seeds
    random.seed(args.seed_num)
    np.random.seed(args.seed_num)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)

    n_gpu = torch.cuda.device_count()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # Get output directories
    # wandb_name = f"Base_L{num_layers}_emb{num_embed_dim}_SL{max_input_size}_E{epochs}_B{genecompass_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_O{optimizer}_DS{num_gpus}"
    training_output_dir = f"{args.output_directory}/models/{args.run_name}/"
    logging_dir = f"{args.output_directory}/runs/{args.run_name}/"
    model_output_dir = os.path.join(training_output_dir, "models/")

    # Ensure not overwriting previously saved model
    model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
    if os.path.isfile(model_output_file) is True:
        raise Exception("Model already saved to this directory.")

    # Make training and model output directories
    if args.world_rank == 0:
        subprocess.call(f"mkdir -p {training_output_dir}", shell=True)
        subprocess.call(f"mkdir -p {model_output_dir}", shell=True)

    # Load gene_id and prior knowledge:token dictionary 
    with open(args.token_dict_path, "rb") as fp:
        token_dictionary = pickle.load(fp)

    knowledges = dict()
    out = load_prior_embedding(token_dictionary_or_path=args.token_dict_path)
    knowledges['promoter'] = out[0]
    knowledges['co_exp'] = out[1]
    knowledges['gene_family'] = out[2]
    knowledges['peca_grn'] = out[3]
    knowledges['homologous_gene_human2mouse'] = out[4]

    # Model configuration
    # Same as the size of BERT base
    config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "attention_probs_dropout_prob": 0.02,
        "hidden_dropout_prob": 0.02,
        "intermediate_size": 3072,
        "hidden_act": "gelu",
        "max_position_embeddings": 2048,
        "model_type": "bert",
        "num_attention_heads": 12,
        "pad_token_id": token_dictionary.get("<pad>"),
        "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
        "use_values": True,
        "use_promoter": True,
        "use_co_exp": True,
        "use_gene_family": True,
        "use_peca_grn": True,
        "warmup_steps": args.warmup_steps,
        "emb_warmup_steps": args.emb_warmup_steps,
        "warmup_steps": args.warmup_steps,
        "emb_warmup_steps": args.emb_warmup_steps,
        "use_cls_token": True,
    }

    model_config = BertConfig(**config)
    model = BertForMaskedLM(model_config, knowledges=knowledges)
    # Set to the training mode
    model = model.train()

    # Define the training arguments of Huggingface trainer
    training_args = {
        "run_name": args.run_name,
        "fp16": args.fp16,     # auto mixed precision
        "fp16_opt_level": "O1",   # fp16 level
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing": args.gradient_checkpointing,   # lower gpu memory cost
        # "report_to": 'none',  # online/offline wandb report
        "dataloader_num_workers": args.dataloader_num_workers,
        "learning_rate": args.max_learning_rate,
        "do_train": args.do_train,
        "do_eval": args.do_eval,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.train_micro_batch_size_per_gpu,
        # "max_steps": max_steps,
        "num_train_epochs": args.num_train_epochs, 
        "save_strategy": "steps" if args.save_model else None,
        "save_steps": args.save_steps if args.save_model else None,
        "logging_steps": 100,
        "output_dir": training_output_dir,
        "logging_dir": logging_dir,
    }
    training_args = TrainingArguments(**training_args)

    # Load training dataset
    train_dataset = load_from_disk(args.dataset_directory)
    example_lengths_file = os.path.join(args.dataset_directory, 'sorted_length.pickle')

    if args.world_rank == 0:
        print("Starting training.")

    # Define the Huggingface trainer
    trainer = GenecompassPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        example_lengths_file=example_lengths_file,
        token_dictionary=token_dictionary,
    )

    # Start training
    trainer.train()
    # ckpt_path = '/home/share/genecompass_github/xCompass/pretrained_models/GeneCompass_Base'
    # trainer.train(resume_from_checkpoint=ckpt_path)

    # save model
    if args.save_model:
        trainer.save_model(model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None, type=str,)
    parser.add_argument('--seed_num', type=int, default=0,)
    parser.add_argument('--seed_val', type=int, default=42,)
    parser.add_argument("--token_dict_path", default=None, type=str, required=True,)
    parser.add_argument("--dataset_directory", default=None, type=str, required=True,)
    parser.add_argument("--num_train_epochs", default=5, type=int,)
    parser.add_argument("--train_micro_batch_size_per_gpu", default=10, type=int,)
    parser.add_argument("--eval_micro_batch_size_per_gpu", default=10, type=int,)
    parser.add_argument("--max_learning_rate", default=5e-5, type=float,)
    parser.add_argument("--min_learning_rate", default=0.0, type=float,)
    parser.add_argument("--warmup_steps", default=10000, type=int,)
    parser.add_argument("--emb_warmup_steps", default=10000, type=int,)
    parser.add_argument("--lr_scheduler_type", default="linear", type=str,)
    parser.add_argument("--weight_decay", default=0.01, type=float,)
    parser.add_argument("--dataloader_num_workers", default=0, type=int,)
    parser.add_argument("--output_directory", default=None, type=str, required=True,)
    parser.add_argument("--do_train", action="store_true",)
    parser.add_argument("--do_eval", action="store_true",)
    parser.add_argument("--eval_strategy", default="epoch", type=str, help="Do eval loop by epoch or the whole task")
    parser.add_argument("--eval_steps", default=100000, type=int,)
    parser.add_argument("--save_model", action="store_true",)
    parser.add_argument("--save_strategy", default="steps", type=str, help="epoch or steps.")
    parser.add_argument("--save_steps", default=100000, type=int,)
    parser.add_argument("--local-rank", type=int, default=-1,)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,)
    parser.add_argument("--fp16", action="store_true",)
    parser.add_argument('--gamma', type=float, default=0.5,)
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",)

    args = parser.parse_args()

    main(args)