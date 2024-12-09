import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

passage_dataset = load_dataset("llm-book/aio-passages", split="train")

# モデルの初期化
model_path = "src/q_and_a/bpr/outputs_bpr/passage_encoder"
tokenizer = AutoTokenizer.from_pretrained(model_path)
passage_encoder = AutoModel.from_pretrained(model_path)

device = "cuda:0"
passage_encoder = passage_encoder.to(device)


def embed_passages(titles: list[str], texts: list[str]) -> np.ndarray:
    """パッセージの埋め込みを取得する"""

    # パッセージのトークン化
    tokenized_passages = tokenizer(
        titles,
        texts,
        padding=True,
        truncation="only_second",
        max_length=256,
        return_tensors="pt",
    ).to(device)

    # パッセージの実数埋め込みを取得
    with torch.inference_mode():
        with torch.amp.autocast():
            encoded_passages = passage_encoder(**tokenized_passages).last_hidden_state[
                :, 0
            ]

    # 実数埋め込みをnumpy配列に変換
    emb = encoded_passages.cpu().numpy()
    emb = np.where(emb < 0, 0, 1).astype(bool)
    emb = np.packbits(emb).reshape(emb.shape[0], -1)

    return emb


if __name__ == "__main__":
    print("Embedding passages...")
    passage_dataset = passage_dataset.map(
        lambda x: {
            "embeddings": list(
                embed_passages(
                    x["title"],
                    x["text"],
                )
            )
        },
        batched=True,
    )

    passage_dataset.save_to_disk("src/q_and_a/bpr/outputs_bpr/embedded_passages")
