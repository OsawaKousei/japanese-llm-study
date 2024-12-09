import logging
import random
from logging import Formatter, StreamHandler, getLogger
from pprint import pprint

import torch
from datasets import load_dataset
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
train_dataset = load_dataset("llm-book/aio-retriever", split="train")
logger.info("train_dataset: %s", train_dataset)
pprint(train_dataset[0])

valid_dataset = load_dataset("llm-book/aio-retriever", split="validation")
logger.info("valid_dataset: %s", valid_dataset)
pprint(valid_dataset[0])

# 正例とハード負例を持たないデータを削除
train_dataset = train_dataset.filter(
    lambda x: (
        len(x["positive_passage_indices"]) > 0
        and len(x["negative_passage_indices"]) > 0
    )
)
valid_dataset = valid_dataset.filter(
    lambda x: (
        len(x["positive_passage_indices"]) > 0
        and len(x["negative_passage_indices"]) > 0
    )
)


def filter_train_passages(example: dict) -> dict:
    """最も関連度の高いpassageを取得する"""
    example["positive_passage_indices"] = [example["positive_passage_indices"][0]]
    return example


train_dataset = train_dataset.map(filter_train_passages)


def filter_valid_passages(example: dict) -> dict:
    """最も関連度の高いpassageを取得する"""
    example["positive_passage_indices"] = [example["positive_passage_indices"][0]]
    example["negative_passage_indices"] = [example["negative_passage_indices"][0]]
    return example


valid_dataset = valid_dataset.map(filter_valid_passages)

# トークナイザの読み込み
base_model_names = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(base_model_names)


# collate_fnの定義
def collate_fn(examples: list[dict]) -> dict[str, BatchEncoding | Tensor]:
    """BPRの訓練・検証データのミニバッチ作成"""
    questions: list[str] = []
    passage_titles: list[str] = []
    passage_texts: list[str] = []

    for example in examples:
        questions.append(example["question"])

        # 正例とハード負例のpassageを取得
        positive_passage_idx = random.choice(example["positive_passage_indices"])
        negative_passage_idx = random.choice(example["negative_passage_indices"])

        passage_titles.extend(
            [
                example["passages"][positive_passage_idx]["title"],
                example["passages"][negative_passage_idx]["title"],
            ]
        )

        passage_texts.extend(
            [
                example["passages"][positive_passage_idx]["text"],
                example["passages"][positive_passage_idx]["text"],
            ]
        )

    # トークナイズ
    tokenized_questions = tokenizer(
        questions, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    tokenized_passages = tokenizer(
        passage_titles,
        passage_texts,
        padding=True,
        truncation="only_second",
        max_length=256,
        return_tensors="pt",
    )

    # 正例の位置を示すTensorを作成
    labels = torch.arange(0, len(questions) * 2, 2)

    return {
        "tokenized_questions": tokenized_questions,
        "tokenized_passages": tokenized_passages,
        "labels": labels,
    }
