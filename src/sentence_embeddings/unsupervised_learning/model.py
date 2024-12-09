import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, BatchEncoding
from transformers.utils import ModelOutput


class SimCSEModel(nn.Module):
    """SimCSEモデル"""

    def __init__(
        self,
        base_model_name: str,
        mlp_only_train: bool = False,
        temperature: float = 0.05,
    ):
        super().__init__()

        # エンコーダーの初期化
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # MLP層の次元数
        self.hidden_size = self.encoder.config.hidden_size
        # MLP層の線形層
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # 活性化関数
        self.activation = nn.Tanh()

        # MLP層による変換を訓練時にのみ適用するよう設定するフラグ
        self.mlp_only_train = mlp_only_train
        # 温度パラメータ
        self.temperature = temperature

    def encode_texts(self, tokenized_texts: BatchEncoding) -> Tensor:
        """エンコーダーを用いて文をベクトルに変換する"""
        encoded_texts = self.encoder(**tokenized_texts)

        # [CLS]トークンの特徴量を取得
        encoded_texts = encoded_texts.last_hidden_state[:, 0, :]

        if self.mlp_only_train and not self.training:
            return encoded_texts

        # MLP層による変換
        encoded_texts = self.dense(encoded_texts)
        encoded_texts = self.activation(encoded_texts)

        return encoded_texts

    def forward(
        self,
        tokenized_text_1: BatchEncoding,
        tokenized_text_2: BatchEncoding,
        labels: Tensor,
        label_scores: Tensor | None = None,
    ) -> ModelOutput:
        """SimCSEの順伝播処理"""

        # 文をベクトルに変換
        encoded_text_1 = self.encode_texts(tokenized_text_1)
        encoded_text_2 = self.encode_texts(tokenized_text_2)

        # コサイン類似度を計算
        sim_matrix = F.cosine_similarity(
            encoded_text_1.unsqueeze(1), encoded_text_2.unsqueeze(0), dim=2
        )

        # 損失
        loss = F.cross_entropy(sim_matrix / self.temperature, labels)

        positive_mask = F.one_hot(labels, num_classes=sim_matrix.size(1)).bool()
        positive_scores = torch.masked_select(sim_matrix, positive_mask)

        return ModelOutput(
            loss=loss,
            scores=positive_scores,
        )


base_model_name = "cl-tohoku/bert-base-japanese-v3"
unsup_model = SimCSEModel(base_model_name=base_model_name, mlp_only_train=True)
