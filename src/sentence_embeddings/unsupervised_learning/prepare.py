import logging
from logging import Formatter, StreamHandler, getLogger

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
unsup_train_dataset = load_dataset("llm-book/jawiki-sentences", split="train")
logger.debug("unsup_train_dataset: %s", unsup_train_dataset)

# データセットのサンプルを表示
for i, text in enumerate(unsup_train_dataset[:50]["text"]):
    logger.debug(i, text)

# 訓練セットから空行を削除
unsup_train_dataset = unsup_train_dataset.filter(lambda x: x["text"].strip() != "")

# 訓練セットをシャッフルし、最初の100万行を使用
unsup_train_dataset = unsup_train_dataset.shuffle().select(range(1000000))
# ディスクに書き込む
unsup_train_dataset = unsup_train_dataset.flatten_indices()
logger.debug("sampled 1000000 unsup_train_dataset: %s", unsup_train_dataset)

# データセットのサンプルを表示
for i, text in enumerate(unsup_train_dataset[:10]["text"]):
    logger.debug(i, text)

# JSTSデータセットの読み込み
valid_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="train")
test_dataset = load_dataset("llm-book/JGLUE", name="JSTS", split="validation")
logger.debug("valid_dataset: %s", valid_dataset)
logger.debug("test_dataset: %s", test_dataset)

# モデルのトークナイザーの読み込み
base_model_name = "cl-tohoku/bert-base-japanese-v3"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
logger.debug("tokenizer: %s", tokenizer)


def unsup_train_collate_fn(examples: list[dict]) -> dict[str, BatchEncoding | Tensor]:
    """教師なしsimCSEの訓練データセットのミニバッチを作成する関数"""

    # トークナイザーを適用
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # 類似度行列における正例ペアの位置を示すTensorを作成
    labels = torch.arange(len(examples))

    return {
        "tokenized_text_1": tokenized_texts,
        "tokenized_text_2": tokenized_texts,
        "labels": labels,
    }


def eval_collate_fn(examples: list[dict]) -> dict[str, BatchEncoding | Tensor]:
    """教師なしsimCSEの評価データセットのミニバッチを作成する関数"""

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
