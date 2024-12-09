import logging
import random
from logging import Formatter, StreamHandler, getLogger
from pprint import pprint
from typing import Iterator

import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from transformers import AutoTokenizer, BatchEncoding
from transformers.trainer_utils import set_seed

# ログの設定
if __name__ == "__main__":
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler_format = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
else:
    logger = getLogger("__main__").getChild(__name__)

# 乱数シードの設定
set_seed(42)

# データセットの読み込み
jsnli_dataset = load_dataset("llm-book/jsnli", split="train")
logger.debug("unsup_train_dataset: %s", jsnli_dataset)
pprint(jsnli_dataset[0])
pprint(jsnli_dataset[1])

# 前提文とラベルごとに仮設文をまとめたdictを作成
premises2hypotheses: dict[str, dict[str, list[str]]] = {}

premises = jsnli_dataset["premise"]
hypotheses = jsnli_dataset["hypothesis"]
labels = jsnli_dataset["label"]

for premise, hypothesis, label in zip(premises, hypotheses, labels):
    if premise not in premises2hypotheses:
        premises2hypotheses[premise] = {
            "entailment": [],
            "neutral": [],
            "contradiction": [],
        }

    premises2hypotheses[premise][label].append(hypothesis)


def generate_sup_train_example() -> Iterator[dict[str, str]]:
    """教師あり学習の訓練データセットのサンプルを生成する関数"""

    for premise, hypotheses in premises2hypotheses.items():
        if len(hypotheses["contradiction"]) == 0:
            continue

        for entailment_hypothesis in hypotheses["entailment"]:
            contradiction_hypothesis = random.choice(hypotheses["contradiction"])

            yield {
                "premise": premise,
                "entailment_hypothesis": entailment_hypothesis,
                "contradiction_hypothesis": contradiction_hypothesis,
            }


sup_train_dataset = Dataset.from_generator(generate_sup_train_example)
logger.debug("sup_train_dataset: %s", sup_train_dataset)
pprint(sup_train_dataset[0])
pprint(sup_train_dataset[1])


# JSTSデータセットの読み込み
valid_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="train")
test_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="validation")
logger.debug("valid_dataset: %s", valid_dataset)
logger.debug("test_dataset: %s", test_dataset)

# モデルのトークナイザーの読み込み
base_model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
logger.debug("tokenizer: %s", tokenizer)


def sup_train_collate_fn(examples: list[dict]) -> dict[str, BatchEncoding | Tensor]:
    """教師ありsimCSEの訓練データセットのミニバッチを作成する関数"""

    premises = []
    hypotheses = []

    for example in examples:
        premises.append(example["premise"])

        entailment_hypothesis = example["entailment_hypothesis"]
        contradiction_hypothesis = example["contradiction_hypothesis"]

        hypotheses.extend([entailment_hypothesis, contradiction_hypothesis])

    # トークナイザーを適用
    tokenized_premises = tokenizer(
        premises,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    tokenized_hypotheses = tokenizer(
        hypotheses,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # 類似度行列における正例ペアの位置を示すTensorを作成
    labels = torch.arange(0, 2 * len(examples), 2)

    return {
        "tokenized_text_1": tokenized_premises,
        "tokenized_text_2": tokenized_hypotheses,
        "labels": labels,
    }


def eval_collate_fn(examples: list[dict]) -> dict[str, BatchEncoding | Tensor]:
    """教師ありsimCSEの評価データセットのミニバッチを作成する関数
    教師なしと同じ関数を使う
    """

    # トークナイザーを適用
    tokenized_texts1 = tokenizer(
        [example["sentence1"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    tokenized_texts2 = tokenizer(
        [example["sentence2"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    labels = torch.arange(len(examples))

    # データセットに付与された類似度スコアのTensorを作成
    label_scores = torch.tensor([example["label"] for example in examples])

    return {
        "tokenized_text_1": tokenized_texts1,
        "tokenized_text_2": tokenized_texts2,
        "labels": labels,
        "label_scores": label_scores,
    }
