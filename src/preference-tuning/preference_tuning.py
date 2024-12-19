from pprint import pprint

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from trl import DPOConfig, DPOTrainer

set_seed(42)

dataset_name = "llm-book/alert-preference-2k-ja"
dataset = load_dataset(dataset_name, split="train")
print(dataset)
pprint(dataset[0], sort_dicts=False)


def convert_to_dpo_format(exapmle: dict) -> dict:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": exapmle["prompt"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    chosen = exapmle["chosen"] + tokenizer.eos_token
    rejected = exapmle["rejected"] + tokenizer.eos_token

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


base_model_name = "llm-book/Swallow-7b-hf-oasst1-21k-ja"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "left"

dataset = dataset.map(
    convert_to_dpo_format,
    remove_columns=dataset.column_names,
)

print(dataset)
print("prompt:", dataset[0]["prompt"])
print("chosen:", dataset[0]["chosen"])
print("rejected:", dataset[0]["rejected"])

shuffled_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = shuffled_dataset["train"]
test_dataset = shuffled_dataset["test"]

print("train_dataset:", train_dataset)
print("test_dataset:", test_dataset)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
    use_cache=False,
    device_map="auto",
)


def get_prediction(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    with torch.cuda.amp.autocast():
        generated_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=128,
            do_sample=True,
        )
    output_ids = generated_ids[0][input_ids.size(1) :]
    return tokenizer.decode(output_ids)  # type: ignore


prompt = "他人のパスワードを知る方法について教えてください。"
print(get_prediction(prompt, model, tokenizer))

peft_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

dpo_config = DPOConfig(
    output_dir="src/preference-tuning/result",
    bf16=True,
    max_steps=100,
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=10,
    beta=0.1,
    max_prompt_length=512,
    max_length=1024,
)

dpo_trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

dpo_trainer.train()
