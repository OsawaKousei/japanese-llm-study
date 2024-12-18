import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM

dataset = load_dataset("llm-book/oasst1-21k-ja", split="train")
base_model_name = "tokyotech-llm/Swallow-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.chat_template = """\
{%- for message in messages %}
{%- if message['role'] == 'user' %}
{{ bos_token + 'ユーザ：' + message['content'] + eos_token }}
{%- elif message['role'] == 'assistant' %}
{{ bos_token + 'アシスタント：'  + message['content'] + eos_token }}
{%- endif %}
{% if loop.last and add_generation_prompt %}
{{ bos_token + 'アシスタント：' }}
{%- endif %}
{% endfor %}\
"""
tokenized_dataset = [
    tokenizer.apply_chat_template(item["conversation"]) for item in dataset
]
tokenizer.pad_token = tokenizer.unk_token
bos = tokenizer.bos_token
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=bos + "ユーザ：",
    response_template=bos + "アシスタント：",
    tokenizer=tokenizer,
)

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

model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="src/instruction_tuning/IT_results",
    bf16=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=300,
    report_to="none",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    data_collator=collator,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()
