# Introduction
Currently the tasks include: [Gene-expression-profiling](https://github.com/pth1993/DeepCE)(Based on DeepCE model)


# Installion
Please use conda to create your environment, since we have different models and methods, we recommand to create different envirionments for each cases, here are .yml files as shown in below:


For model GeneCompass(including generate embedding), testing Gene-expression-profiling task:


```
# Please confirm prefix in GeneCompass.yaml first !!
# if you have completed it in GRN inference task, you do not need to install it again, just activate it !!

conda env create -f GeneCompass.yaml
```

# Pretrained-weights

The Pretrained-weights are provided in "./model/GeneCompass"


# How to use this project
## 1. Generate embedding


### 1.1 Generate embedding from GeneCompass
Please run command as shown in below

```
python get_emb.py
```

### 1.2 Generate embedding from your own model
It depends on different situations, if you get embedding file from yourselves, please place them under folder "tasks"

## 2. Test performence of large model on Gene-expression-profiling task

Before you test on this task, please generate embedding from model

After you get the embedding, please run command as shown in below

```
# activate GeneCompass environment


python get_result-Gene-expression-profiling-GeneCompass.py    # For GeneCompass
```

## Dependency files

The dependency files can be found at [GRN_Inference_Files](https://drive.google.com/drive/folders/1v3fDJ5hWwFMjnYLLdzE39LDshGutj3Ts?usp=sharing).

# References

For Gene-expression-profiling task, we refer DeepCE model to complete our jobs, the paper is [A deep learning framework for high-throughput mechanism-driven phenotype compound screening and its application to COVID-19 drug repurposing](https://www.nature.com/articles/s42256-020-00285-9). The code is at:(https://github.com/pth1993/DeepCE)
