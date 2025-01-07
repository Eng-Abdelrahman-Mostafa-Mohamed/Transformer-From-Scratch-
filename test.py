import torch
import torch.nn as nn
import math
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from textblob import TextBlob as tb

# 1. Embedding Layer
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x)

# 2. Positional Encoding Layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 3. Multi-Head Attention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % heads == 0, 'd_model must be divisible by heads'
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wout = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))  # Apply mask to prevent attention to padding tokens
        attention = torch.softmax(scores, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        return torch.matmul(attention, value)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        def transform(x, linear):
            return linear(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        query = transform(q, self.Wq)
        key = transform(k, self.Wk)
        value = transform(v, self.Wv)

        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.Wout(x)

# 4. Position-wise Feedforward Layer
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

# 5. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Multi-head attention
        attn_output = self.attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))
        # Feedforward layer
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))
        return x

# 6. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn1 = MultiHeadAttention(d_model, heads, dropout)
        self.attn2 = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Multi-head self-attention
        attn_output1 = self.attn1(x, x, x, tgt_mask)
        x = self.layernorm1(x + self.dropout1(attn_output1))
        # Multi-head attention with encoder output
        attn_output2 = self.attn2(x, encoder_output, encoder_output, src_mask)
        x = self.layernorm2(x + self.dropout2(attn_output2))
        # Feedforward layer
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output))
        return x

# 7. Encoder
class Encoder(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, N: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 8. Decoder
class Decoder(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, N: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x

# 9. Projection Layer
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)

# 10. Transformer Model
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # Encoding the source input
        encoder_output = self.encode(src, src_mask)
        # Decoding the target input using the encoder output
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # Projecting the decoder output to the vocabulary space
        return self.project(decoder_output)

# 11. Helper Functions for Masking
def create_padding_mask(seq, pad_token_id):
    return (seq != pad_token_id).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask == 0  # Returns a lower triangular matrix of False/True

# 12. Training Function
def train_transformer(config, data):
    data = process_data(data)
    training_data = CreateTrainingDataForTransformer(config, data)
    training_data_loader = DataLoader(training_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

    src_vocab_size = len(training_data.src_tokenizer.get_vocab())
    tgt_vocab_size = len(training_data.tgt_tokenizer.get_vocab())

    model = build_transformer(
        src_seq_len=get_max_seq_len(data),
        tgt_seq_len=get_max_seq_len(data),
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=512, h=8, N=6, dropout=0.1, d_ff=2048
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=training_data.tgt_tokenizer.token_to_id("<pad>"))

    for epoch in range(10):
        for src_tokenized, tgt_tokenized in training_data_loader:
            optimizer.zero_grad()
            tgt_input = tgt_tokenized[:, :-1]
            tgt_output = tgt_tokenized[:, 1:]
            
            # Generate padding masks for source and target sequences
            src_mask = create_padding_mask(src_tokenized, training_data.pad_token_id)
            tgt_mask = create_padding_mask(tgt_input, training_data.pad_token_id) & create_look_ahead_mask(tgt_input.size(1))

            # Forward pass
            output = model(src_tokenized, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
