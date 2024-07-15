# Introduction
Currently the tasks include: [Drug_dose_response](https://github.com/theislab/cpa#readme)(Predicting cellular responses to different types of drugs and their dosages, based on Compositional Perturbation Autoencoder (CPA) model)


# Installion

For testing Drug_dose_response task:

Please use conda to create your environment, since we have different models and methods, we recommand to create different envirionments for each cases, here are .yml files as shown in below:


```
# Please confirm prefix in cpa.yaml first !!

conda env create -f cpa.yaml
```


# Pretrained-weights

The Pretrained-weights are provided in "./model/GeneCompass"


# How to use this project
## 1. Generate embedding


### 1.1 Generate embedding from GeneCompass
Please run command as shown in below

```
# when you running this program, please make sure you have installed envirionments in GeneCompass.yaml, and activate it !!
python get_emb.py
```

### 1.2 Generate embedding from your own model
It depends on different situations, if you get embedding file from yourselves, please place them under folder "drug_dose_response"

## 2. Test performence of large model on Dose-Response task
Before you test on this task, please generate embedding from model

After you get the embedding, please run command as shown in below

```
# please activate cpa environment first !!


python get_result-Dose-Response-GeneCompass.py    # For GeneCompass
```


# References
For Drug dose response task, we refer CPA model to complete our jobs, the paper is [Learning interpretable cellular responses to complex perturbations in high-throughput screens](https://www.biorxiv.org/content/10.1101/2021.04.14.439903v2). The code is at:(https://github.com/facebookresearch/CPA)