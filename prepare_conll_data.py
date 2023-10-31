import os
from os import listdir, path
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

source = "training_subtask_en.csv"
dest = 'output/'

### =========================================
### Writing the contents into the output file
### =========================================
def write_conll_data(conll_data, file_path, splits):
    data = {'train': [], 'test': [], 'dev': []}
    train_split = splits.get('train', [])
    test_split = splits.get('test', [])
    dev_split = splits.get('dev', [])

    for ind, line in enumerate(conll_data):
        for label_list in line:
            if ind in train_split:
                data['train'].append(label_list[0] + "\t" + label_list[1])
            elif ind in test_split:
                data['test'].append(label_list[0] + "\t" + label_list[1])
            else:
                data['dev'].append(label_list[0] + "\t" + label_list[1])
        if ind in train_split:
            data['train'].append('')
        elif ind in test_split:
            data['test'].append('')
        else:
            data['dev'].append('')

    for split_type, value_list in splits.items():
        with open(file_path + split_type + ".txt", 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(data[split_type]))

def write_instruction_data(instruction_data, file_path, splits):
    for split_type, value_list in splits.items():
        with open(file_path + split_type + ".json", 'w', encoding='utf-8') as fp:
            json.dump([val for ind, val in enumerate(instruction_data) if ind in value_list], fp, indent=4)

### =========================================

### Reading the pandas csv dataset
df = pd.read_csv(source, delimiter=';')

### Generating the dataset format
conll_data = []
instruction_data = []
for i in tqdm(range(df.shape[0])):

    label_data = []

    text = df.iloc[i]['Text'].strip()
    cause_text = df.iloc[i]['Cause'].strip()
    effect_text = df.iloc[i]['Effect'].strip()

    cause_start = text.find(cause_text)
    cause_end = cause_start + len(cause_text)

    effect_start = text.find(effect_text)
    effect_end = effect_start + len(effect_text)


    ### ==========================================
    ### Preparing instruction dataset (start)
    ### ==========================================
    entry = {
        'context': text,
        'entity_values': {},
        'entity_spans': []
    }
    if cause_text!='':
        entry['entity_values']['Cause'] = [cause_text]
        entry['entity_spans'].append({
            'start': cause_start,
            'end': cause_end,
            'label': 'Cause'
        })
    if effect_text!='':
        entry['entity_values']['Effect'] = [effect_text]
        entry['entity_spans'].append({
            'start': effect_start,
            'end': effect_end,
            'label': 'Effect'
        })
    instruction_data.append(entry)
    ### ==========================================
    ### Preparing instruction dataset (end)
    ### ==========================================


    ### ==========================================
    ### Preparing conll dataset (start)
    ### ==========================================
    text = text.replace(cause_text, ' <cause> ')
    text = text.replace(effect_text, ' <effect> ')

    total_tokens = [tok for tok in text.split(' ') if tok.strip()!='']
    cause_tokens = [tok for tok in cause_text.split(' ') if tok.strip()!='']
    effect_tokens = [tok for tok in effect_text.split(' ') if tok.strip()!='']


    for tok in total_tokens:
        if tok == "<cause>":
            for ind, c_tok in enumerate(cause_tokens):
                if ind==0:
                    label_data.append([c_tok, 'B-Cause'])
                else:
                    label_data.append([c_tok, 'I-Cause'])

        elif tok == "<effect>":
            for ind, c_tok in enumerate(effect_tokens):
                if ind==0:
                    label_data.append([c_tok, 'B-Effect'])
                else:
                    label_data.append([c_tok, 'I-Effect'])

        else:
            label_data.append([tok, 'O'])

    conll_data.append(label_data)
    ### ==========================================
    ### Preparing conll dataset (end)
    ### ==========================================


assert len(conll_data)==len(instruction_data)

train_split, test_split = train_test_split(range(len(conll_data)), test_size=0.05, shuffle=True, random_state=42)
train_split, dev_split = train_test_split(train_split, test_size=0.05, shuffle=True, random_state=42)
print(len(train_split), len(dev_split), len(test_split))

### writing into files
write_conll_data(conll_data, dest, {'train': train_split, 'dev': dev_split, 'test': test_split})
write_instruction_data(instruction_data, dest, {'train': train_split, 'dev': dev_split, 'test': test_split})

### sample prining
print(conll_data[5])
print(instruction_data[5])