import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from textblob import TextBlob as tb
from Transformer import Transformer, build_transformer
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset
import arabic_reshaper
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import json
# from json import 

config = {
    'tokenizer_path': '.',
    'model_path': '.',
    'data_path': './ara.csv',
    'src_lang': 'ar',
    'tgt_lang': 'en'
}


def get_all_sents(ds,lang):
    for itm in ds:
        yield itm['translation'][lang]

def build_tokenizer(config, data, lang):
    tokenizer_path = Path(config['tokenizer_path']) / f"tokenizer_{lang}.json"
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.WordLevel(unk_token="[<unk>]")) # replace any word not in the vocabulary with <unk>
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # split the text into words using white spaces
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[<unk>]", "[<start>]", "[<end>]", "[<pad>]"], min_frequency=2 # <start> and <end> tokens are used to indicate the beginning and end of a sentence respectively and <pad> is used to pad the input sequences to the same length during training 
        )
        tokenizer.train_from_iterator(get_all_sents(data,lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset_from_hugging_face(config):
    train_data = load_dataset('Helsinki-NLP/opus-100',f"{config['src_lang']}-{config['tgt_lang']}",split='train')
    test_data = load_dataset('Helsinki-NLP/opus-100',f"{config['src_lang']}-{config['tgt_lang']}",split='test')
    validation_data = load_dataset('Helsinki-NLP/opus-100',f"{config['src_lang']}-{config['tgt_lang']}",split='validation')
    
    train_src_tokenizer = build_tokenizer(config, train_data, config['src_lang'])
    train_tgt_tokenizer = build_tokenizer(config, train_data, config['tgt_lang'])
    
    
    return train_data, test_data, validation_data

class CreateTrainingDataForTransformer(Dataset):
    def __init__(self, config, data , tokenizer_src, tokenizer_tgt,seq_len):
        self.config = config
        self.data = data
        self.seq_len = seq_len
        self.src_tokenizer = tokenizer_src
        self.tgt_tokenizer = tokenizer_tgt
        
        self.sos_token = self.src_tokenizer.token_to_id("[<start>]")
        self.eos_token = self.src_tokenizer.token_to_id("[<end>]")
        self.pad_token = self.src_tokenizer.token_to_id("[<pad>]")
        
        self.src_text = data['translation'][config['src_lang']]
        self.tgt_text = data['translation'][config['tgt_lang']]
        
        self.src_tokenizer = build_tokenizer(config, data, config['src_lang'])
        self.tgt_tokenizer = build_tokenizer(config, data, config['tgt_lang'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt_of_idx= self.src_text[idx]
        tgt_txt_of_idx = self.tgt_text[idx]
        
        src_tokens_input_enc = self.src_tokenizer.encode(src_txt_of_idx).ids
        tgt_tokens_input_dec = self.tgt_tokenizer.encode(tgt_txt_of_idx).ids
        
        src_to_enc_num_padding_needed = self.seq_len - len(src_tokens_input_enc)
        tgt_dec_num_padding_needed = self.seq_len - len(tgt_tokens_input_dec)
        
        if src_to_enc_num_padding_needed <0 or tgt_dec_num_padding_needed <0:
            raise ValueError("The sentense input too long")
        
        
        #encoder input i want to make it be like this <sos_token> <src_tokens_input_enc>  <pad_tokens>=<pad_tokens> redundant N=src_to_enc_num_padding_needed
        
        
        """ Hint Important note : so in training we train on current input tokens to encoder "src lang"

            and internaly we train on prev seq on sec lang using encoder output and previous tokens of decoder input (tgt language ) {THE KEY AND VALUE INPUT TO CROSS ATTENTION BLOCK IN DECODER } 
            
        """  
            
        encoder_input = torch.cat( torch.tensor(self.sos_token), 
                                   
                                   torch.tensor(src_tokens_input_enc , dtype=torch.int64),
                                   
                                   torch.tensor(self.eos_token , dtype=torch.int64),
                                   
                                   torch.tensor([self.pad_token]*src_to_enc_num_padding_needed , dtype=torch.int64)
                                   
                                   

                                  )
        
        # the input of decoder is using for training the model to predict the target token in training time  itnwill be 
        decoder_input = torch.cat( 
                            torch.tensor(self.sos_token),
                            
                            torch.tensor(tgt_tokens_input_dec , dtype=torch.int64),
                            
                            torch.tensor([self.pad_token]*tgt_dec_num_padding_needed , dtype=torch.int64),
                            

                            )
        
        lable = torch.cat(
                            torch.tensor(self.sos_token), 
                                   
                            torch.tensor(tgt_tokens_input_dec , dtype=torch.int64),
                                   
                            torch.tensor([self.pad_token]*tgt_dec_num_padding_needed , dtype=torch.int64),
                                   
                            torch.tensor(self.eos_token , dtype=torch.int64)

        )
        
        
        # to check that padding operation applied correctly
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert lable.size(0) == self.seq_len        
                                   
        return {
            'encoder_input': encoder_input, # the dimention now is seq length
            'decoder_input': decoder_input,
            'lable': lable,
            #but we want the padded tokens doesn't affect the attention mechanism so we need to create a mask for the encoder and decoder
                                                                                #first for batch_size  , second for seq_length and third for the seq_length (we want to reshape mask to work with self attention scores to apply multiblications on  it with the scores) on it  
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).type(torch.int64).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'src_txt': src_txt_of_idx,
            'tgt_txt': tgt_txt_of_idx
        }
        
        
def causal_mask(tgt_seq_len):
    mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len))
    return mask == 0


def process_data(data):
    data = data.apply(lambda x: x.astype(str).str.lower())
    data = data.apply(lambda x: x.astype(str).str.replace(r'[^\w\s]', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\d+', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\n', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\r', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\s+', ' ', regex=True))
    # data = data.apply(lambda x: str(tb(x).correct()))
    return data


def get_max_seq_len(data):
    max_seq_len = 0
    for i in range(len(data)):
        src_text = data[config['src_lang']][i]
        tgt_text = data[config['tgt_lang']][i]
        src_tokenized = len(src_text.split())
        tgt_tokenized = len(tgt_text.split())
        if src_tokenized > max_seq_len:
            max_seq_len = src_tokenized
        if tgt_tokenized > max_seq_len:
            max_seq_len = tgt_tokenized
    return max_seq_len


def train_transformer(config, data):
    data = process_data(data)
    training_data = CreateTrainingDataForTransformer(config, data)
    training_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

    src_vocab_size = len(training_data.src_tokenizer.get_vocab())
    tgt_vocab_size = len(training_data.tgt_tokenizer.get_vocab())

    model = build_transformer(
        src_seq_len=get_max_seq_len(data[config['src_lang']]),
        tgt_seq_len=get_max_seq_len(data[config['tgt_lang']]),
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
            output = model(src_tokenized, tgt_input)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')


if __name__ == '__main__':
    train_data , test_data, validation_data = get_dataset_from_hugging_face(config)
    print(len(train_data), len(test_data), len(validation_data))
print(f"the train data ex_check -- Arabic-- {arabic_reshaper.reshape(train_data[:5]['translation']['ar'][0])} -- English-- {train_data[:5]['en'][0]}")
print(f"the validation data ex_check -- Arabic-- {arabic_reshaper.reshape(validation_data[:5]['ar'][0])} -- English-- {validation_data[:5]['en'][0]}")
print(f"the test data ex_check -- Arabic-- {arabic_reshaper.reshape(test_data[:5]['ar'][0])} -- English-- {test_data[:5]['en'][0]}")