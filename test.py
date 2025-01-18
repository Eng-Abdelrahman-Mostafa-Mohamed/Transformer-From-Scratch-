import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
# from textblob import TextBlob as tb
from Transformer import build_transformer
from torch.utils.data import Dataset, DataLoader 
from datasets import load_dataset
# import arabic_reshaper
# from tqdm import tqdm
from Transformer import load_model , train_transformer , get_Transformer

model = load_model(config, len(src_tokenizer.get_vocab()), len(tgt_tokenizer.get_vocab()), config['model_path'])


# Translate a sentence
sentence = "أهلاً بك في عالم الذكاء الاصطناعي"
translated_sentence = translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, config['seq_len'])
print(translated_sentence)