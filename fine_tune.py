import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

def main():



    # Load dataset from the hub
    target_lang = 'kor'
    print(target_lang)
    language_map = {
        'kor':'Korean',
        'tel':'Telugu'
    }
    english = load_dataset("gsarti/flores_101",'eng')
    target = load_dataset("gsarti/flores_101",target_lang)
    train_data = english['dev'].add_column("sentence2",target['dev']['sentence'])

    print(f'dataset size: {len(train_data)}')
    prefix = f'Translate English to {language_map[target_lang]}: '
    answer = 'Answer: '

    def preprocessing(examples):
        inputs = [prefix + s1+' '+ answer + s2 for s1, s2 in zip(examples['sentence'], examples['sentence2'])] 
        return inputs
    train_data = train_data.map(preprocessing)