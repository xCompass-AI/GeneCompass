#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python ../DeepCE/main_deepce.py --drug_file "../DeepCE/data/drugs_smiles.csv" \
--gene_file "../DeepCE/data/gene_vector.csv"  --train_file "../DeepCE/data/signature_train.csv" \
--dev_file "../DeepCE/data/signature_dev.csv" --test_file "../DeepCE/data/signature_test.csv" \
--dropout 0.1 --batch_size 16 --max_epoch 1 --emb_file "../../GeneCompass-emb.npy" > ../DeepCE/output/output.txt