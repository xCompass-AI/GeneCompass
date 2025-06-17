# Introduction
Currently the tasks include: [GRN](https://github.com/HantaoShu/DeepSEM/tree/master)(Gene Regulatory Network, based on DeepSEM-master)


# Installion
Please use conda to create your environment, since we have different models and methods, we recommand to create different envirionments for each cases, here are .yml files as shown in below:

For model GeneCompass(including generate embedding)

```
# Please confirm prefix in GeneCompass.yaml first !!

conda env create -f GeneCompass.yaml
```

# Pretrained-weights

The Pretrained-weights are provided in "./model/GeneCompass"

## Dependency files

The dependency file can be found at [GRN_Inference_Files](https://drive.google.com/drive/folders/1v3fDJ5hWwFMjnYLLdzE39LDshGutj3Ts?usp=sharing).

# How to use this project

## Test performence of large model on GRN task
Please run command as shown in below 

```
python get_result-GRN-GeneCompass.py    # For GeneCompass, activate GeneCompass environment
```

# References
For GRN task, we refer DeepSEM framework to complete our jobs, the paper is [Modeling gene regulatory networks using neural network architectures](https://www.nature.com/articles/s43588-021-00099-8). The code is at:(https://github.com/HantaoShu/DeepSEM)
   

