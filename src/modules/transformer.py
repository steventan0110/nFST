import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.functional import F
import math
from transformers import GPT2LMHeadModel, GPT2Config


class GPT2Wrapper(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.bos = args.bos
        self.eos = args.eos
        self.pad = args.pad
        self.max_length = args.max_length
        self.num_layers = args.tilde_p_num_layers
        self.num_heads = args.num_heads
        self.config = GPT2Config(
            args.vocab_size,
            n_head=args.num_heads,
            n_layer=args.tilde_p_num_layers,
            bos_token_id=self.bos,
            eos_token_id=self.eos,
            n_positions=args.max_length + 1,
            n_embd=args.tilde_p_hid_dim,
        )
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, x):
        """
        x: input tokens of (bz, seq_len)
        """
        # shift by 1 on x
        bz, seq_len = x.shape
        pad_token = x.new_full((bz, 1), self.pad)
        gold_x = torch.cat((x, pad_token), dim=1)
        bos_token = x.new_full((bz, 1), self.bos)
        shift_x_as_input = torch.cat((bos_token, x), dim=1)

        logits = self.model(shift_x_as_input).logits  # bz x seq x vocab_size
        logits[:, :, self.pad] = -float(1e8)  # -infinity for pad token
        logits = F.log_softmax(logits, dim=2)

        log_prob = torch.gather(logits, 2, gold_x.unsqueeze(2)).squeeze(2)
        pad_mask = (~(gold_x == self.pad)).long()  # bz x seq
        log_prob = log_prob * pad_mask
        return torch.sum(log_prob, dim=1)


class PositionalEncoding(nn.Module):
    """
    Classic Attention-is-all-you-need positional encoding.
    From PyTorch docs.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int):
    """Generate a triangular (size, size) mask. From PyTorch docs."""
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, nheads, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, nheads, max_len)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MultiheadAttention(nn.Module):
    """my own batch-first implementation of multihead attention"""

    def __init__(self, embed_dim, num_heads, max_len):
        super().__init__()
        self.num_heads = num_heads
        assert (
            embed_dim % self.num_heads == 0
        ), "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
        self.register_buffer(
            "mask", torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        )
        self.reset_params()

    def reset_params(self):
        xavier_normal_(self.key.weight)
        xavier_normal_(self.value.weight)
        xavier_normal_(self.query.weight)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, embed_dim)
        k_t = (
            self.key(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.value(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .reshape(batch_size, seq_len, self.num_heads, -1)
            .transpose(1, 2)
        )
        # shape == (batch_size, num_heads, seq_len, head_dim)

        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))

        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, float("-1e8"))
        attn = self.attn_dropout(attn)

        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v)

        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, embed_dim)
        y = self.proj_dropout(self.proj(y))

        return y


class TransformerLM(nn.Module):
    """
    Classic Transformer that both encodes and decodes.

    Prediction-time inference is done greedily.

    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(
        self,
        num_classes: int,
        max_output_length: int,
        dim: int,
        bos,
        pad,
        eos,
        num_layers,
        num_heads,
    ):
        super().__init__()

        # Parameters
        self.dim = dim
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.max_output_length = max_output_length
        self.step = 0

        # Encoder part
        self.embedding = nn.Embedding(num_classes, dim)
        self.pos_encoder = PositionalEncoding(d_model=self.dim)
        self.transformer_decoder = nn.Sequential(
            *[
                DecoderBlock(dim, num_heads, max_output_length)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(self.dim)
        self.fc = nn.Linear(self.dim, num_classes)

        # It is empirically important to initialize weights properly
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, Sx) with elements in (0, C) where C is num_classes
        Output: (B, Sy, C) logits as well as a score based on force-decoding (B,)
        """

        pad_mask = (~(input == self.pad)).long()
        x = self.embedding(input) * math.sqrt(self.dim)  # (B, Sx, E)
        x = self.pos_encoder(x)  # (B, Sx, E)
        x = self.transformer_decoder(x)  # (B, Sx, E)
        x = self.ln(x)
        x = self.fc(x)  # bz x seq x Vocab
        x = F.log_softmax(x, dim=2)
        log_prob = torch.gather(x, 2, input.unsqueeze(2)).squeeze(2)  # bz x seq
        log_prob = log_prob * pad_mask  # apply mask to remove pad logits
        # if self.step % 100 == 0:
        #     print("print out some transformer output")
        #     print(x[0, 5, :])
        #     print(torch.max(x[0, 5, :]))
        #     print(log_prob[0, 5])

        self.step += 1
        # apply mask
        out = torch.sum(log_prob, dim=1)  # bz
        return out


class Transformer(nn.Module):
    """
    Classic Transformer that both encodes and decodes.

    Prediction-time inference is done greedily.

    NOTE: start token is hard-coded to be 0, end token to be 1. If changing, update predict() accordingly.
    """

    def __init__(self, num_classes: int, max_output_length: int, dim: int = 256):
        super().__init__()

        # Parameters
        self.dim = dim
        self.max_output_length = max_output_length
        nhead = 8
        num_layers = 6
        dim_feedforward = dim

        # Encoder part
        self.embedding = nn.Embedding(num_classes, dim)
        self.pos_encoder = PositionalEncoding(d_model=self.dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        )

        # Decoder part
        self.y_mask = generate_square_subsequent_mask(self.max_output_length)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.dim, nhead=nhead, dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(self.dim, num_classes)

        # It is empirically important to initialize weights properly
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)  # (Sx, B, E)
        output = self.decode(y, encoded_x)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (Sx, B, E) embedding
        """
        x = x.permute(1, 0)  # (Sx, B)
        x = self.embedding(x) * math.sqrt(self.dim)  # (Sx, B, E)
        x = self.pos_encoder(x)  # (Sx, B, E)
        x = self.transformer_encoder(x)  # (Sx, B, E)
        return x

    def decode(self, y: torch.Tensor, encoded_x: torch.Tensor) -> torch.Tensor:
        """
        Input
            encoded_x: (Sx, B, E)
            y: (B, Sy) with elements in (0, C) where C is num_classes
        Output
            (Sy, B, C) logits
        """
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(encoded_x)  # (Sy, Sy)
        output = self.transformer_decoder(y, encoded_x, y_mask)  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to use at inference time. Predict y from x one token at a time. This method is greedy
        decoding. Beam search can be used instead for a potential accuracy boost.

        Input
            x: (B, Sx) with elements in (0, C) where C is num_classes
        Output
            (B, C, Sy) logits
        """
        encoded_x = self.encode(x)

        output_tokens = (
            (torch.ones((x.shape[0], self.max_output_length))).type_as(x).long()
        )  # (B, max_length)
        output_tokens[:, 0] = 0  # Set start token
        for Sy in range(1, self.max_output_length):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1:]  # Set the last output token
        return output_tokens
