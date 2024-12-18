import itertools
from pprint import pprint

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM

set_seed(42)

dataset = load_dataset("llm-book/oasst1-21k-ja", split="train")

if __name__ == "__main__":
    print(dataset)
    pprint(dataset[0])

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

if __name__ == "__main__":
    chat_text = tokenizer.apply_chat_template(
        dataset[0]["conversation"], tokenize=False
    )
    print(chat_text.replace(tokenizer.eos_token, "\n"))

tokenized_dataset = [
    tokenizer.apply_chat_template(item["conversation"]) for item in dataset
]

if __name__ == "__main__":
    token_ids = tokenized_dataset[0]
    print("token_ids:", token_ids)
    print("tokens:", tokenizer.convert_ids_to_tokens(token_ids))

tokenizer.pad_token = tokenizer.unk_token

bos = tokenizer.bos_token
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=bos + "ユーザ：",
    response_template=bos + "アシスタント：",
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    batch = collator(tokenized_dataset[:1])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("入力トークン:", input_ids)
    print("正解ラベル:", labels)

    segments_to_fit: list[list[int]] = []
    segments_to_ignore: list[list[int]] = []

    for key, group in itertools.groupby(
        range(len(input_ids)), key=lambda i: labels[i] == -100
    ):
        group_list = list(group)
        if key:
            segments_to_ignore.append(group_list)
        else:
            segments_to_fit.append(group_list)

        print("----損失を計算しない部分----")
        for seg in segments_to_ignore:
            print(tokenizer.decode(input_ids[seg]))
            print()

        print("----損失を計算する部分----")
        for seg in segments_to_fit:
            print(tokenizer.decode(input_ids[seg]))
            print()
