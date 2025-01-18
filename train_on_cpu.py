import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from textblob import TextBlob as tb
from Transformer import build_transformer
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset
import arabic_reshaper
from tqdm import tqdm
from Transformer import load_model , train_transformer , get_Transformer

# torch.cuda.set_device(0)
# print('-----------------------------------------------------------------')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device) 
print(torch.device('cpu')) 
print('-----------------------------------------------------------------')
config = {
    'tokenizer_path': '.',
    'model_path': 'traiend_model/best_model_weights.pth',
    'src_lang': 'ar',
    'tgt_lang': 'en',
    'batch_size': 1,
    'seq_len': 6670,
    'd_model': 512,
    'N' : 1,
    'h' : 8,
    'dropout' : 0.1,
    'd_ff' : 2048,
    'lr' : 0.01,
}

def get_all_sents(ds, lang):
    for itm in ds:
        yield itm['translation'][lang]

def build_tokenizer(config, data, lang):
    tokenizer_path = Path(config['tokenizer_path']) / f"tokenizer_{lang}.json"
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(models.WordLevel(unk_token="[<unk>]")) # models.WordLevel(unk_token="[<unk>]" iterate on words and replace un known word with <unk> token and making tokenizer 
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # tokinizing  using spaces 
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[<unk>]", "[<start>]", "[<end>]", "[<pad>]"], min_frequency=2 # min_frequency=2 means that the word that is repeated 2 times or more will be tokenized
        )
        tokenizer.train_from_iterator(get_all_sents(data, lang), trainer=trainer) # training the tokenizer on the data
        tokenizer.save(str(tokenizer_path))  
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset_from_hugging_face(config):
    train_data = load_dataset('Helsinki-NLP/opus-100', f"{config['src_lang']}-{config['tgt_lang']}", split='train')
    test_data = load_dataset('Helsinki-NLP/opus-100', f"{config['src_lang']}-{config['tgt_lang']}", split='test')
    print(f"{train_data[1]['translation'][config['src_lang']]}")
    validation_data = load_dataset('Helsinki-NLP/opus-100', f"{config['src_lang']}-{config['tgt_lang']}", split='validation')
    
    train_src_tokenizer = build_tokenizer(config, train_data, config['src_lang'])
    train_tgt_tokenizer = build_tokenizer(config, train_data, config['tgt_lang'])
    
    ready_train_data = CreateTrainingDataForTransformer(train_data, train_src_tokenizer, train_tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    ready_val_data = CreateTrainingDataForTransformer(validation_data, train_src_tokenizer, train_tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    
    max_src_seq_len = 0
    max_tgt_seq_len = 0
    for item in train_data:
        max_src_seq_len = max(max_src_seq_len, len(train_src_tokenizer.encode(item['translation'][config['src_lang']]).ids))
        max_tgt_seq_len = max(max_tgt_seq_len, len(train_src_tokenizer.encode(item['translation'][config['tgt_lang']]).ids))
        
    print(f"max_src_seq_len {max_src_seq_len} max_tgt_seq_len {max_tgt_seq_len}")
    
    train_data_loader = DataLoader(ready_train_data, batch_size=config['batch_size'], shuffle=True)
    val_data_loader = DataLoader(ready_val_data, batch_size=1, shuffle=True)
    
    return train_data_loader, val_data_loader, train_src_tokenizer, train_tgt_tokenizer, test_data

class CreateTrainingDataForTransformer(Dataset):
    def __init__(self, data, tokenizer_src, tokenizer_tgt, src_language, target_language, seq_len):
        self.config = config
        self.data = data
        self.seq_len = seq_len
        self.src_tokenizer = tokenizer_src
        self.tgt_tokenizer = tokenizer_tgt
        self.src_language = src_language
        self.target_language = target_language
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[<start>]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[<end>]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[<pad>]")])
        self.src_tokenizer = build_tokenizer(config, data, config['src_lang'])
        self.tgt_tokenizer = build_tokenizer(config, data, config['tgt_lang'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt_of_idx = self.data[idx]['translation'][self.src_language]
        tgt_txt_of_idx = self.data[idx]['translation'][self.target_language]
        
        src_tokens_input_enc = torch.tensor((self.src_tokenizer.encode(src_txt_of_idx).ids),dtype=torch.int64)
        tgt_tokens_input_dec = torch.tensor((self.tgt_tokenizer.encode(tgt_txt_of_idx).ids),dtype=torch.int64)
        
        src_to_enc_num_padding_needed = (self.seq_len - len(src_tokens_input_enc) - 2)
        tgt_dec_num_padding_needed = (self.seq_len - len(tgt_tokens_input_dec) - 1)
        
        if src_to_enc_num_padding_needed < 0 or tgt_dec_num_padding_needed < 0:
            raise ValueError("The sentence input is too long")
        
        # Hint : the encoder input should be [sos] + src_tokens_input_enc + [eos] + [pad] * num_padding_needed 
        # Hint : the decoder input should be [sos] + tgt_tokens_input_dec + [pad] * num_padding_needed why we dont add [eos] token here? 
        # Hint : the lable should be tgt_tokens_input_dec + [eos] + [pad] * num_padding_needed
        # the reson is   Shifting the target sequence by one position to the right, the decoder input should be [sos] + tgt_tokens_input_dec + [pad] * num_padding_needed
        
        # my name is abdelrahman
        # [sos] my name is abdelrahman [eos]
        # Target --> [sos] ich bin abdelrahman shifted by one position to the right decoder could see the current and previous token only and predect next token 
        # lable --> ich bin abdelrahman [eos]   # the lable should be the target shifted by one position to the right
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(src_tokens_input_enc, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * src_to_enc_num_padding_needed, dtype=torch.int64)
        ],dim=0)
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(tgt_tokens_input_dec, dtype=torch.int64),
            torch.tensor([self.pad_token] * tgt_dec_num_padding_needed, dtype=torch.int64)
        ],dim=0)
        
        lable = torch.cat([
            torch.tensor(tgt_tokens_input_dec, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * tgt_dec_num_padding_needed, dtype=torch.int64),
        ],dim=0)
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len # chack that its padded
        assert lable.size(0) == self.seq_len
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'lable': lable,
            'encoder_mask': ((encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()),
            'decoder_mask': ((decoder_input != self.pad_token).type(torch.int64).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))),
            'src_txt': src_txt_of_idx,
            'tgt_txt': tgt_txt_of_idx
        }

def causal_mask(tgt_seq_len):
    mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len))
    return mask == 0

# date is ready no need to preprocess but its important step to preprocess the data
# def process_data(data):
#     data = data.apply(lambda x: x.astype(str).str.lower())
#     data = data.apply(lambda x: x.astype(str).str.replace(r'[^\w\s]', '', regex=True)) # its used for removing special characters from the text data 
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\d+', '', regex=True)) # its used for removing digits from the text data
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\n', '', regex=True)) # its used for removing new line from the text data
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\r', '', regex=True)) # its used for removing carriage return from the text data
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\s+', ' ', regex=True))
#     data = data.apply(lambda x: str(tb(x).correct()))
#     return data



def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_seq_len):
    model.eval()
    
    # Tokenize the input sentence
    sentence = arabic_reshaper.reshape(sentence)
    tokens = src_tokenizer.encode(sentence).ids
    tokens = [src_tokenizer.token_to_id("[<start>]")] + tokens + [src_tokenizer.token_to_id("[<end>]")]
    tokens += [src_tokenizer.token_to_id("[<pad>]")] * (max_seq_len - len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0)
    
    
    # Create the input mask
    src_mask = (tokens != src_tokenizer.token_to_id("[<pad>]")).unsqueeze(1).unsqueeze(2).int()
    
    
    # Generate predictions
    output = model.generate(tokens, max_length=max_seq_len, src_mask=src_mask)
    
    # Decode the outputs
    output_tokens = output.squeeze().tolist()
    translated_sentence = tgt_tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    return translated_sentence

if __name__ == '__main__':
    
    train_data_loader, val_data_loader, src_tokenizer, tgt_tokenizer, test_data = get_dataset_from_hugging_face(config)
    epochs = 10
    
    # the main problem of training the transformer is cuda out of memory error so we mostly Fine tune the model on the hugging face
    model = train_transformer(config, train_data_loader, val_data_loader, len(src_tokenizer.get_vocab()), len(tgt_tokenizer.get_vocab()), epochs,config['lr'])
    
    model = load_model(config, len(src_tokenizer.get_vocab()), len(tgt_tokenizer.get_vocab()), config['model_path'])

    # Translate a sentence
    sentence = "أهلاً بك في عالم الذكاء الاصطناعي"
    translated_sentence = translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, config['seq_len'])
    print(translated_sentence)