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


### ================================== Helper Functions ==============================================
def get_predictions(text, model, tokenizer, label2id, id2label):
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
    
    return prediction_str

### ================================== Helper Functions ==============================================


def main()
    ### ================================== Parameters ==============================================
    model_checkpoint = "./model/"
    os.environ["WANDB_DISABLED"] = "true"

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


    ### ================================== Load model ==============================================
    config = PeftConfig.from_pretrained(model_checkpoint, token = access_token)
    inference_model = AutoModelForTokenClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(inference_model, model_checkpoint)
    ### ================================== Load model ==============================================

    text = input("Enter Financial Document Text: ")
    print("Predictions: ", get_predictions(text, model, tokenizer, label2id, id2label))

if __name__ == "__main__":
    main()