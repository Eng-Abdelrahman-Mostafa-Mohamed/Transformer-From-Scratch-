import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', message='not allowed')

class Input_Embedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int):
        super .__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embeddings = nn.Embeddings(vocab_size,d_model)
    def forward(self,x):
        return self.embeddings(x) * torch.sqrt(d_model)




class Positional_Encoding(nn.Module):
    def __init__(self, seq_length:int,d_model:int,dropout:float):
        super .__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.dropout = nn.Dropout()
        self.pe = torch.zeros(seq_length,d_model)  # word1 ---> em1     (s * d) 
                                                   # word2  -->em 2
        self.position = torch.arrange(0,seq_length,torch.float).unsqueez(1)
        self.div_term = torch.exp(torch.arrange(0,seq_length,torch.float,2) * (-torch.log(10000.0) / d_model))
        #apply sin for odd and cos for even i in embedding vector
        self.pe[:,0::2] = torch.sin(self.position * self.div_term)
        self.pe[:,1::2] = torch.cos(self.position * self.div_term)
        self.pe = self.pe.unsqueez(1) # patch   seq  dim
        
        self.register_buffer(pe)
    def forward(self,x):
        x = x + self.pe[:,x.shape[1],:].requires_grad(False)
        return x



class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10 ** -7):
        
        #eps is a small number to avoid division by zero
        #alpha and gamma are learnable parameters
        #alpha and gamma control influence of normalized value and original value
        
        super .__init__()
        self.eps=eps
        self.alpha = nn.parameter(torch.ones(1)) 
        self.gamma = nn.parameter(torch.ones(1))
    def forward(self,x):
        self.mu = x.mean(dim=-1 , keepdim=True)
        self.sigma = x.std(dim=-1 , keepdim=True)
        return alpha*(x-mu)/(sigma+eps)*self.gamma


class feed_forrward_nn(nn.Module):
    def __init__(self,d_model,_hid_:int,dropout:float):
        super .__init__()
        self.lin1 = nn.Linear(d_model,_hid_)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(_hid_,d_model)
    def forward(self,normalized_x):
        
        return self.lin2(self.dropout(torch.relu(self.lin1(normalized_x))))


class multi_head_atten(nn.Module):
    def __init__(self,d_model:int,heads:int,dropout:float)->None:
        super .__init__()
        self.d_model = d_model
        self.h = heads
        assert d_model % heads == 0 , 'd_model couldnt devide into number of attention heads '
        self.d_k = d_model // self.heads
        
        Wq =  nn.linear(d_model,d_model) # number of nurons of current layer equal d model
        Wk =  nn.linear(d_model,d_model)
        Wv =  nn.linear(d_model,d_model)


        Wout =  nn.linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    
    @staticmethod
    def attention(self,query,key,value,mask,dropout=nn.Dropout):
        #query(batch,h,seq,dk) @ (batch,h,dk,seq) --> (batch,h,seq,seq)
        attention_scores = (query @ key.Transpose(-2,-1)) / torch.sqrt(self.d_k)
        if mask != None : # mask ==0 mean that is padded seqvocab
            attention_scores.fill_mask(mask==0,-1e9)
        attention_scores.Softmax(dim=-1) #(batch,h,seq,seq)

        if dropout != None:
            attention_scores = dropout(attention_scores)
        # now we returning scores for visualization and attentionscores @ values for creating attentioned embeddings 
        return (attention_scores @ value) , attention_scores
        
    def forward(self,q,k,v,mask): #(batch,seq_length,d_model)
        query = Wq(q)
        key = WK(k)
        value = Wv(v)
        # we want to partitioning for multy heads so new dimention that 
        query = torch.view(query.shape[0],query.shape[1],h,d_k).Transpose(1,2) #we want dmodel-> h * dk --> h * (seq * dk) batch h seq dk batch h   
        key = torch.view(key.shape[0],key.shape[1],h,d_k).Transpose(1,2)
        query = torch.view(value.shape[0],value.shape[1],h,d_k).Transpose(1,2)

        x, self.attention_scores = self.attention(query,key,value,mask,self.dropout)
           #batch seq h d_k   *  batch seq h d_k   ==> batch seq h dk  
        x=x.Transpose(1,2).contiguous().view(x.shape[0],-1,self.h * self.d_k) 
        return self.Wout(x) # batch srq_length  d_model


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super() .__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

class projection_layer(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super .__init__()
        self.lin = nn.Linear(d_model,vocab_size)
    def forward(self,x):
        return torch.log_softmax(self.lin(x),dim=-1)


class EncoderBlock(nn.Module):
                    # same input used for query and key and value 
    def _init_(self, self_attention_block: multi_head_atten, feed_forward_block: feed_forrward_nn, dropout: float) -> None:
        super()._init_()
        
        #for each encoder we have 2 residual_connections || 1 self_attention || 1fedd forrward nn  
        # first resdiual block compines input embedding and normalized version of output of m.H.A  block 
        # second compines output from first resdinual block and output of FFNN block 
        
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


# the encoder is a stack of N encoder blocks so we use nn.ModuleList to stack them
class Encoder(nn.Module):
    def _init_(self, layers: nn.ModuleList) -> None:
        super()._init_()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):                               #cross attention block
    def _init_(self, self_attention_block: multi_head_atten, src_attention_block: multi_head_atten, feed_forward_block: feed_forrward_nn, dropout: float) -> None:
        super()._init_()
        self.self_attention_block = self_attention_block
        self.src_attention_block = src_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x,encoder_output,src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.src_attention_block(x,encoder_output , encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

    # Hint that the decoder is a stack of N decoder blocks too
    #src_mask is used to mask the source sequence (the source language that we want translate from)
    #tgt_mask is used to mask the target sequence (the target language that we want translate to)
    #we will see when go to training part that we pass target language emeddings into self_attention_block and the output of encoder into src_attention_block( so it called cross attention we cross compine 2 parts together [encoder and decoder]) 


class Decoder(nn.Module):
    def _init_(self, layers: nn.ModuleList) -> None:
        super()._init_()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def _init_(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, tgt_embed: nn.Module, output: projection_layer) -> None:
        super()._init_()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.project = output

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt),src_mask, tgt_mask)
    
    
    
    # def generate(self, x):
    #     return self.project(x)

    