import logging
from logging import Formatter, StreamHandler, getLogger

import faiss
from datasets import Dataset
from text_embedding import embed_texts
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

emb_dim = encorder.config.hidden_size
index = faiss.IndexFlatIP(emb_dim)

# ベクトルの読み込み
dataset_path = (
    "src/sentence_embeddings/nearest_neighbor_search/outputs/paragraph_embeddings"
)
paragraph_dataset = Dataset.load_from_disk(dataset_path)
# インデックスの追加
paragraph_dataset.add_faiss_index(column="embeddings")

query_text = "アメリカ合衆国の首都"

# 最近傍探索
scores, retrieved_examples = paragraph_dataset.get_nearest_examples(
    "embeddings", embed_texts([query_text])[0], k=10
)

# 結果の表示
titles = retrieved_examples["title"]
texts = retrieved_examples["text"]
for socre, title, text in zip(scores, titles, texts):
    logger.info("score: %s, title: %s, text: %s", socre, title, text)
