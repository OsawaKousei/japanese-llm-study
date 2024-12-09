import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # Q: (batch_size, n_heads, seq_len)
        # K: (batch_size, n_heads, seq_len)
        # V: (batch_size, n_heads, seq_len)
        # mask: (batch_size, 1, seq_len)
        scaler = np.sqrt(self.d_k)
        attention_weight = (
            torch.matmul(Q, K.transpose(1, 2)) / scaler
        )  # 「Q * X^T / (D^0.5)」

        if mask is not None:
            if mask.dim() != attention_weight.dim():
                raise ValueError(
                    "mask.dim != attention_weight.dim, mask.dim={}, attention_weight.dim={}".format(
                        mask.dim(), attention_weight.dim()
                    )
                )
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )

        attention_weight = nn.functional.softmax(
            attention_weight, dim=2
        )  # Attention weightを計算
        return torch.matmul(
            attention_weight, V
        )  # (Attention weight) * X により重み付け.


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super.__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.W_k = nn.Parameter(
            torch.Tensor(
                n_heads, d_model, self.d_k
            )  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_q = nn.Parameter(
            torch.Tensor(
                n_heads, d_model, self.d_k
            )  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.W_v = nn.Parameter(
            torch.Tensor(
                n_heads, d_model, self.d_v
            )  # ヘッド数, 入力次元, 出力次元(=入力次元/ヘッド数)
        )

        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_3d: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len = q.size(0), q.size(1)

        """repeat Query,Key,Value by num of heads"""
        q = q.repeat(self.n_heads, 1, 1, 1)  # head, batch_size, seq_len, d_model
        k = k.repeat(self.n_heads, 1, 1, 1)  # head, batch_size, seq_len, d_model
        v = v.repeat(self.n_heads, 1, 1, 1)  # head, batch_size, seq_len, d_model

        """Linear before scaled dot product attention"""
        q = torch.einsum(
            "hijk,hkl->hijl", (q, self.W_q)
        )  # head, batch_size, d_k, seq_len
        k = torch.einsum(
            "hijk,hkl->hijl", (k, self.W_k)
        )  # head, batch_size, d_k, seq_len
        v = torch.einsum(
            "hijk,hkl->hijl", (v, self.W_v)
        )  # head, batch_size, d_k, seq_len

        """Split heads"""
        q = q.view(self.h * batch_size, seq_len, self.d_k)
        k = k.view(self.h * batch_size, seq_len, self.d_k)
        v = v.view(self.h * batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        """Scaled dot product attention"""
        attention_output = self.scaled_dot_product_attention(
            q, k, v, mask_3d
        )  # (head*batch_size, seq_len, d_model)

        attention_output = torch.chunk(attention_output, self.n_heads, dim=0)
        attention_output = torch.cat(attention_output, dim=2)

        """Linear after scaled dot product attention"""
        output = self.linear(attention_output)
        return output


class AddPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.d_model))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.d_model + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(relu(self.linear1(x)))


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_idx: int = 0) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_idx
        )

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        return self.embedding_layer(input_batch)
