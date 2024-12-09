import logging
from logging import Formatter, StreamHandler, getLogger
from pprint import pprint

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

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

# モデルの読み込み
model_path = (
    "src/sentence_embeddings/unsupervised_learning/outputs_unsup_simcse/encoder"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
encorder = AutoModel.from_pretrained(model_path)

device = "cuda:0"
encorder.to(device)


def embed_texts(texts: list[str]) -> np.ndarray:
    """文をベクトルに変換する"""
    tokenized_texts = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        encoded_texts = encorder(**tokenized_texts).last_hidden_state[:, 0]

    emb = encoded_texts.cpu().numpy().astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return emb


if __name__ == "__main__":

    # データセットの読み込み
    paragraph_dataset = load_dataset("llm-book/jawiki-paragraphs", split="train")
    logger.info(paragraph_dataset)
    pprint(paragraph_dataset[0])
    pprint(paragraph_dataset[1])

    # 各記事の最初の段落を取得
    paragraph_dataset = paragraph_dataset.filter(lambda x: x["paragraph_index"] == 0)
    logger.info(paragraph_dataset)
    pprint(paragraph_dataset[0])
    pprint(paragraph_dataset[1])

    device = "cuda:0"
    encorder.to(device)

    paragraph_dataset = paragraph_dataset.map(
        lambda x: {"embeddings": list(embed_texts(x["text"]))},
        batched=True,
    )
    logger.info(paragraph_dataset)
    pprint(paragraph_dataset[0])

    # ベクトルの保存
    paragraph_dataset.save_to_disk("outputs/paragraph_embeddings")
