#!/usr/bin/env bash

python ../DeepCE/main_drug_repurposing.py --data_dir '../DeepCE/data/covid_data' --patient_file 'covid_signature_ngdc.csv' \
--num_cell 5 --top 100