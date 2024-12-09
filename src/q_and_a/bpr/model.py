import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, BatchEncoding
from transformers.utils import ModelOutput


class BPRModel(nn.Module):
    """BPRモデル"""

    def __init__(self, base_model_name: str):
        super().__init__()

        # 質問エンコーダ
        self.question_encoder = AutoModel.from_pretrained(base_model_name)
        # パッセージエンコーダ
        self.passage_encoder = AutoModel.from_pretrained(base_model_name)
        # モデルの訓練ステップ数
        self.global_step = 0

    def binary_encode(self, x: Tensor) -> Tensor:
        """BPRのためのバイナリエンコーディング"""

        if self.training:
            return torch.tanh(x * math.pow((1.0 + self.global_step * 0.1), 0.5))
        else:
            return torch.where(x >= 0, 1.0, -1.0).to(x.device)

    def encord_questions(
        self, tokenized_questions: BatchEncoding
    ) -> tuple[Tensor, Tensor]:
        """質問を実数埋め込みとバイナリ埋め込みにエンコード"""
        encoded_questions = self.question_encoder(
            **tokenized_questions
        ).last_hidden_state[:, 0]
        binary_encoded_questions = self.binary_encode(encoded_questions)

        return encoded_questions, binary_encoded_questions

    def binary_encode_passages(self, tokenized_passages: BatchEncoding) -> Tensor:
        """パッセージをバイナリエンコード"""
        encoded_passages = self.passage_encoder(**tokenized_passages).last_hidden_state[
            :, 0
        ]
        binary_encoded_passages = self.binary_encode(encoded_passages)

        return binary_encoded_passages

    def compute_loss(
        self,
        encoded_questions: Tensor,
        binary_encoded_questions: Tensor,
        binary_encoded_passages: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """損失関数の計算"""
        num_passages = binary_encoded_passages.size(0)

        # 候補パッセージ生成の損失を計算する
        binary_scores = torch.matmul(
            binary_encoded_questions, binary_encoded_passages.transpose(0, 1)
        )
        positive_mask = F.one_hot(labels, num_classes=num_passages).bool()
        positive_binary_scores = torch.masked_select(
            binary_scores, positive_mask
        ).repeat_interleave(num_passages - 1)
        negative_binary_scores = torch.masked_select(binary_scores, ~positive_mask)
        traget = torch.ones_like(positive_binary_scores).long()
        loss_cand = F.margin_ranking_loss(
            positive_binary_scores, negative_binary_scores, traget, margin=0.1
        )

        # 候補パッセージのリランキングの損失を計算する
        dense_scores = torch.matmul(
            encoded_questions, binary_encoded_passages.transpose(0, 1)
        )
        loss_rerank = F.cross_entropy(
            dense_scores,
            labels,
        )

        loss = loss_cand + loss_rerank

        return loss

    def forward(
        self,
        tokenized_questions: BatchEncoding,
        tokenized_passages: BatchEncoding,
        labels: Tensor,
    ) -> ModelOutput:
        """順伝播"""
        encoded_questions, binary_encoded_questions = self.encord_questions(
            tokenized_questions
        )
        binary_encoded_passages = self.binary_encode_passages(tokenized_passages)

        # 損失の計算
        loss = self.compute_loss(
            encoded_questions, binary_encoded_questions, binary_encoded_passages, labels
        )

        # ステップ数の更新
        if self.training:
            self.global_step += 1

        return ModelOutput(loss=loss)
