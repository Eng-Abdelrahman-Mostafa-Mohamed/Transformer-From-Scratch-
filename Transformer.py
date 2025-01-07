import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.embeddings.embedding_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mu) / (sigma + self.eps) + self.beta

class FeedForwardNN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.lin2(self.dropout(torch.relu(self.lin1(x))))

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

    def attention(self, query, key, value, mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))  # Improved readability
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

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.lin(x), dim=-1)

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNN, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, src_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardNN, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.src_attention_block = src_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.src_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)



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

    # def forward(self, src, tgt, src_mask, tgt_mask):
    #     encoder_output = self.encode(src, src_mask)
    #     decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
    #     return self.project(decoder_output)





def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    encoder_blocks = [EncoderBlock(MultiHeadAttention(d_model, h, dropout), FeedForwardNN(d_model, d_ff, dropout), dropout) for _ in range(N)] # we have N number of encoder blocks that we will pas as nn.moduleList blocks to encoder to make big encoder block 
    decoder_blocks = [DecoderBlock(MultiHeadAttention(d_model, h, dropout), MultiHeadAttention(d_model, h, dropout), FeedForwardNN(d_model, d_ff, dropout), dropout) for _ in range(N)] # same as encoder blocks we have N number of decoder blocks that we will pas as nn.moduleList blocks to decoder to make big decoder block
    
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) # as we know that each model start with Random initialized weights so to avoid time consuming training we are using xavier_uniform_ initializer to make the initialization 
    
    return transformer
