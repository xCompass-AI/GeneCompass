# imports
import os
# Choose the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
import sys
sys.path.append("../../")
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns
sns.set()
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer
from genecompass import BertForSequenceClassification
from transformers.training_args import TrainingArguments
from genecompass import DataCollatorForCellClassification
from genecompass.utils import load_prior_embedding
import argparse
import numpy as np
import random
import torch

token_dictionary_path='./prior_knowledge/human_mouse_tokens.pickle'

# load knowledges
knowledges = dict()
out = load_prior_embedding(token_dictionary_or_path=token_dictionary_path)
knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]

# 加载数据集
data_path = '/mnt/data_sdb/wangx/data/organoid/kidney/merged/'
dataset = load_from_disk(data_path)
dataset = dataset.rename_column("cell_type", "label")

# 划分训练集和测试集
train_test_split = dataset.train_test_split(test_size=0.2, seed=20)
train_set = train_test_split['train']
test_set = train_test_split['test']

# 保存训练集和测试集
dataset_save_path = '/mnt/data_sdb/wangx/data/organoid/kidney/split_datasets/'
os.makedirs(dataset_save_path, exist_ok=True)
train_set.save_to_disk(dataset_save_path + 'train_set')
test_set.save_to_disk(dataset_save_path + 'test_set')

print(f"训练集和测试集已保存到: {dataset_save_path}")
print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")

# 创建标签到 ID 的映射
target_names = set(list(Counter(train_set["label"]).keys()) + list(Counter(test_set["label"]).keys()))
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
print("Label to ID mapping:", target_name_id_dict)

# 保存标签映射
with open(dataset_save_path + 'label_mapping.pickle', 'wb') as f:
    pickle.dump(target_name_id_dict, f)
print(f"标签映射已保存到: {dataset_save_path}label_mapping.pickle")

def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
train_set = train_set.map(classes_to_ids, num_proc=16)
test_set = test_set.map(classes_to_ids, num_proc=16)

# filter dataset for cell types in corresponding training set
trained_labels = list(Counter(train_set['label']).keys())
def if_trained_label(example):
    return example['label'] in trained_labels
test_set = test_set.filter(if_trained_label, num_proc=16)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # calculate accuracy and macro f1 using sklearn's function
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }


# pretrain checkpoint path
checkpoint_path='/home/wangx/code/xCompass/model/GeneCompass_12layers'

# set freeze layer
freeze_layers = 12

# reload pretrained model
model = BertForSequenceClassification.from_pretrained(
    checkpoint_path,
    num_labels=len(target_name_id_dict.keys()),
    output_attentions=False,
    output_hidden_states=False,
    knowledges=knowledges,
)

if freeze_layers > 0:
    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

model = model.to("cuda")
print(model)


# set output dir
output_dir='/mnt/data_sdb/wangx/GeneCompass/cell_anoatation/organoid/kidney/'
# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
training_args = {
    # "run_name": wandb_name,
    "dataloader_num_workers": 2,
    "learning_rate": 5e-5,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": 10,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": "linear",
    "warmup_steps": 100,
    "weight_decay": 0.001,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 30,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
    "metric_for_best_model": "macro_f1",
    "greater_is_better": True,
}
training_args_init = TrainingArguments(**training_args)


# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics
)
# train the cell type classifier
trainer.train()

predictions = trainer.predict(test_set)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)
