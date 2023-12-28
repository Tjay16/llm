from datasets import load_dataset
from random import randrange

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
    def formatting_func(examples):
        return [prefix + s1+'\n'+ answer+ s2 for s1, s2 in zip(examples['sentence'], examples['sentence2'])]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    use_flash_attention = False

    # Hugging Face model id
    #model_id = "NousResearch/Llama-2-7b-hf"  # non-gated
    model_id = "meta-llama/Llama-2-7b-hf" # gated


    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
    )
    model.config.pretraining_tp = 1


    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM", 
    )


    # prepare model for training
    model = prepare_model_for_kbit_training(model)

    from transformers import TrainingArguments
    output_dirs = f"llama_test/{target_lang}"
    args = TrainingArguments(
        output_dir=output_dirs,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        #gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        #logging_steps=10,
        save_strategy="no",
        save_total_limit=2,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,  # disable tqdm since with packing values are in correct
    )


    model = get_peft_model(model, peft_config)

    from trl import SFTTrainer

    max_seq_length = 2048 # max sequence length for model and packing of the dataset

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        formatting_func=formatting_func, 
        #data_collator=
        args=args,

    )

    # train
    trainer.train() # there will not be a progress bar since tqdm is disabled

    # save model
    trainer.save_model()


if __name__ == '__main__':
    main()