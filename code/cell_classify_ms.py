import os

# GPU_NUMBER = [4,5,6,7]
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns

sns.set()
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from geneformer import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
from utils import load_prior_embedding
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--freeze_layers", type=int, default=0)
parser.add_argument("--pretrain_model", type=str, default='humanmouse_100m')
args = parser.parse_args()
freeze_layers = args.freeze_layers
is_train_val_split = args.is_train_val_split
pretrain_model = args.pretrain_model

output_path = './down_stream_outputs/cell_classify/models'
token_dictionary_path = '/data/5_folder/6000W_pretrain_data_merge/human_mouse_tokens.pickle'

# data paths
train_path = "/data/5_folder/Multiple_Sclerosis/Multiple_Sclerosis_6000Wtoken/train_6000W"
test_path = "/data/5_folder/Multiple_Sclerosis/Multiple_Sclerosis_6000Wtoken/test_6000W"

checkpoint_dict={
"humanmouse_100m": '/data/5_folder/checkpoints/BERT-base_new_concat_HM_100M_epoch1_bs10_8_4_lr5e-5_cls_value_4priors_id02value08_L12_emb768_SL2048_E1_B10_LR5e-05_LSlinear_WU10000_Oadamw_DS8/models', 
}
checkpoint_path = checkpoint_dict[pretrain_model]


wandb_name= f"[cell_classify_on_MS_new]Bert_new_concat_{pretrain_model}_4gpu_30epoch_200warmup_5e-5lr_freeze{freeze_layers}"

# load dataset
trainval_set = load_from_disk(train_path)
test_set = load_from_disk(test_path)

# load knowledges
knowledges = dict()
out = load_prior_embedding(token_dictionary_or_path=token_dictionary_path)
knowledges['promoter'] = out[0]
knowledges['co_exp'] = out[1]
knowledges['gene_family'] = out[2]
knowledges['peca_grn'] = out[3]
knowledges['homologous_gene_human2mouse'] = out[4]

# rename columns
trainval_set = trainval_set.rename_column("celltype", "label")
test_set = test_set.rename_column("celltype", "label")

# create dictionary of cell types : label ids
target_names = set(list(Counter(trainval_set["label"]).keys()) + list(Counter(test_set["label"]).keys()))
target_name_id_dict = dict(zip(target_names, [i for i in range(len(target_names))]))
print(target_name_id_dict)

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example
trainval_set = trainval_set.map(classes_to_ids, num_proc=16)
test_set = test_set.map(classes_to_ids, num_proc=16)

# create 90/10 train/eval splits
trainval_set = trainval_set.shuffle(seed=42)
train_set = trainval_set.select([i for i in range(0, round(len(trainval_set) * 0.9))])
val_set = trainval_set.select(
    [i for i in range(round(len(trainval_set) * 0.9), len(trainval_set))])


# filter dataset for cell types in corresponding training set
trained_labels = list(Counter(train_set["label"]).keys())
def if_trained_label(example):
    return example["label"] in trained_labels
val_set = val_set.filter(if_trained_label, num_proc=16)
test_set = test_set.filter(if_trained_label, num_proc=16)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'macro_f1': macro_f1
    }


# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training parameters
# max learning rate
max_lr = 5e-5
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 16
# batch size for training and eval
geneformer_batch_size = 10
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 100
# number of epochs
epochs = 30
# optimizer
optimizer = "adamw"

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

# define output directory path
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}_{current_date.strftime('%X').replace(':','')}"
output_dir = output_path + f"/{datestamp}_{wandb_name}_geneformer_CellClassifier_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training arguments
training_args = {
    "run_name": wandb_name,
    "dataloader_num_workers": 12,
    "learning_rate": max_lr,
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": 10,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": False,
    "lr_scheduler_type": lr_schedule_fn,
    "warmup_steps": warmup_steps,
    "weight_decay": 0.01,
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model=model,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=train_set,
    eval_dataset=val_set,
    compute_metrics=compute_metrics
)
# train the cell type classifier
trainer.train()

# test
predictions = trainer.predict(test_set)
with open(f"{output_dir}predictions.pickle", "wb") as fp:
    pickle.dump(predictions, fp)
trainer.save_metrics("eval", predictions.metrics)
trainer.save_model(output_dir)

# save target_name2id
with open(f"{output_dir}name2id.pickle", "wb") as fp:
    pickle.dump(target_name_id_dict, fp)
