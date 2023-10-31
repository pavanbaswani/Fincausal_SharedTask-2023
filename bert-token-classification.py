from huggingface_hub import notebook_login, login
import os
from tqdm import tqdm
import pandas as pd
import json

from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import evaluate
import torch
import numpy as np
from seqeval.metrics import classification_report


### ================================== Parameters ==============================================
model_checkpoint = "bert-large-cased"
lr = 1e-3
batch_size = 8
num_epochs = 15
os.environ["WANDB_DISABLED"] = "true"
seqeval = evaluate.load("seqeval")

base_url = "data/conll/"
### ================================== Parameters ==============================================


### ================================== Load labels ==============================================
data_dict = {}
with open(base_url + 'label2id.json', 'r') as fp:
    data_dict = json.load(fp)
label2id = {}
for key, value in data_dict.items():
    label2id[key] = int(value)
    
    
data_dict = {}
with open(base_url + 'id2label.json', 'r') as fp:
    data_dict = json.load(fp)
    
id2label = {}
for key, value in data_dict.items():
    id2label[int(key)] = value
### ================================== Load labels ==============================================

### ================================== Helper functions ==============================================
def reformat_data(text_path):
    data = []

    tokens = []
    tags = []
    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                line = line.strip().replace('\n', '').replace('\r', '')
                if line=='':
                    assert len(tokens) == len(tags)
                    flag = False
                    for tag in tags:
                        if tag!=3:
                            flag = True
                            break
                    if flag:
                        data.append({'tokens': tokens, 'tags': tags})
                    tokens = []
                    tags = []
                else:
                    temp = line.split('\t')
                    tokens.append(temp[0])
                    tags.append(label2id.get(str(temp[1]), label2id.get('O')))
    else:
        print("Path: {} do not exists...!".format(text_path))
    
    return data


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_predictions(text):
    chunk_size = 512
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    tokens = inputs.tokens()
    
    predictions = []
    input_ids = inputs["input_ids"]
    token_chunks = [input_ids[:, i:i+chunk_size] for i in range(0, input_ids.shape[1], chunk_size)]
    
    for chunk in token_chunks:
        with torch.no_grad():
            logits = model(chunk).logits
        
        preds = torch.argmax(logits, dim=2)
        predictions.extend(preds[0].tolist())

    prediction_str = []
    for token, prediction in zip(tokens, predictions):
        predicted_tag = model.config.id2label[prediction]
        prediction_str.append(predicted_tag)
#         if not token.startswith("##"):
#             prediction_str.append(predicted_tag)
    
    return prediction_str
### ================================== Helper functions ==============================================


### ================================== Reformat Data ==============================================
train_data = reformat_data(base_url + "train.txt")
print("Length: ", len(train_data))
print("Train sample: ", train_data[0])

val_data = reformat_data(base_url + "dev.txt")
print("Length: ", len(val_data))
print("Val sample: ", val_data[0])

test_data = reformat_data(base_url + "test.txt")
print("Length: ", len(test_data))
print("Test sample: ", test_data[0])
### ================================== Reformat Data ==============================================


### ================================== Saving Data ==============================================
dataset_path = './reformated_data/'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

with open(dataset_path + "train.json", "w", encoding="utf-8") as fp:
    for entry in train_data:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open(dataset_path + "valid.json", "w", encoding="utf-8") as fp:
    for entry in val_data:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
with open(dataset_path + "test.json", "w", encoding="utf-8") as fp:
    for entry in test_data:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

### ================================== Saving Data ==============================================


### ============================= Load Data using Data Loader ========================================
bionlp = load_dataset('json', data_files={'train': 'dataset/train.json', 'valid': 'dataset/valid.json', 'test': 'dataset/test.json'})
print("Sample: ", bionlp["train"][0])

### ============================= Load Data using Data Loader ========================================


### ============================= Loading tokenizer and tokenize data ========================================
label_list = list(label2id.keys())
print("Labels List: ", label_list)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
### ============================= Loading tokenizer and tokenize data ========================================


### ============================= Loading model ========================================
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label_list), id2label=id2label, label2id=label2id
)

peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
)

model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
### ============================= Loading model ========================================


### ============================= Model Arguments and Training ========================================
training_args = TrainingArguments(
    output_dir="bert-large-token-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_bionlp["train"],
    eval_dataset=tokenized_bionlp["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained('./model/')
### ============================= Model Arguments and Training ========================================


### ============================= Model Inference and Testing ========================================
test_data_path = base_url + "test.txt"
test_sents = []
with open(test_data_path, 'r', encoding='utf-8') as fp:
    words = []
    labels = []
    for line in fp.readlines():
        line = line.strip().replace('\n', '').replace('\r', '')
        
        if line=='' or line=='\n':
            test_sents.append({'tokens': words, 'tags': labels})
            words = []
            labels = []
        else:
            word, label = line.split('\t')
            words.append(word)
            labels.append(label)
            
print("# paras: ", len(test_sents))


y_pred = []
y_true = []
for i in tqdm(range(len(test_sents))):
    
    tokens = test_sents[i]['tokens']
    labels = test_sents[i]['tags']
    sentence = ' '.join(tokens)
    preds = get_predictions(sentence)
    y_pred.append(preds)

    y_true.append([])
    for token, label in zip(tokens, labels):
        splits = tokenizer.tokenize(token)
        y_true[i].extend([label for i in range(len(splits))])
    assert len(y_pred[i])==len(y_true[i])
print(classification_report(y_true, y_pred))

### ============================= Model Inference and Testing ========================================