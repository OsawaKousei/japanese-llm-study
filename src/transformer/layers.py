import torch
from basic_layers import FFN, AddPositionalEncoding, Embedding, MultiHeadAttention
from torch import nn
from torch.nn import LayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.layer_norm_self_attention(
            self.__self_attention_block(x, mask) + x
        )  # residual connection, layer norm
        x = self.layer_norm_ffn(
            self.__feed_forward_block(x) + x
        )  # residual connection, layer norm
        return x

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        self attention block
        """
        x = self.multi_head_attention(x, x, x, mask)
        return self.dropout_self_attention(x)  # dropout

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        feed forward block
        """
        return self.dropout_ffn(self.ffn(x))  # dropout


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, pad_idx)

        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                )
                for _ in range(N)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.src_tgt_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_src_tgt_attention = nn.Dropout(dropout_rate)
        self.layer_norm_src_tgt_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        src: torch.Tensor,  # Encoder output
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.layer_norm_self_attention(
            tgt + self.__self_attention_block(tgt, mask_self)
        )

        x = self.layer_norm_src_tgt_attention(
            tgt + self.__src_tgt_attention_block(src, tgt, mask_src_tgt)
        )

        x = self.layer_norm_ffn(x + self.__feed_forward_block(x))

        return x

    def __src_tgt_attention_block(
        self, src: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout_src_tgt_attention(
            self.src_tgt_attention(tgt, src, src, mask)
        )

    def __self_attention_block(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.dropout_self_attention(self.self_attention(x, x, x, mask))

    def __feed_forward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_ffn(self.ffn(x))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(tgt_vocab_size, d_model, pad_idx)
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                )
                for _ in range(N)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,  # Decoder input
        src: torch.Tensor,  # Encoder output
        mask_src_tgt: torch.Tensor,
        mask_self: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(
                tgt,
                src,
                mask_src_tgt,
                mask_self,
            )
        return tgt
