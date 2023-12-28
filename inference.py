from transformers import pipeline
from datasets import load_dataset

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer ,AutoModelForCausalLM
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target',type=str,required=True)
    parser.add_argument('--backbone',action='store_true')
    parser.add_argument('--model',type=str,required=False,default='default')
    args = parser.parse_args()
    #print(args.target)
    language_map = {
        'kor':'Korean',
        'tel':'Telugu',
        'deu':'German'
    }
    # load base LLM model and tokenizer
    if args.backbone:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        model = model.to('cuda')
    else:
        if args.model == 'default':
            output_dir = f"llama_test/{args.target}"
        else:
            output_dir = f"llama_test/{args.model}"
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(output_dir)


    english = load_dataset("gsarti/flores_101",'eng')
    target = load_dataset("gsarti/flores_101",args.target)
    test_data = english['devtest'].add_column("sentence2",target['devtest']['sentence'])
    #print(f'dataset size: {len(train_data)}')
    prefix = f'Translate English to {language_map[args.target]}: '
    answer = 'Answer: '


    for i in range(len(test_data)):
        input = tokenizer.encode(prefix + test_data[i]['sentence']+"\n" + answer,return_tensors='pt').to('cuda')
        outputs = model.generate(
            input_ids=input, 
            max_new_tokens=200, 
            do_sample=True, 
            top_p=0.9,
            temperature=0.9
            )
        print(f"Generated text:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

if __name__ == '__main__':
    main()