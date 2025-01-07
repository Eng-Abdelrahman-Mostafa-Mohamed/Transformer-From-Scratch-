# import pandas as pd
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers
# from pathlib import Path
# import torch
# from torch.utils.data import Dataset, DataLoader
# from textblob import TextBlob as tb
# from Transformer import Transformer, build_transformer


# config = {
#     'tokenizer_path': '.',
#     'model_path': '.',
#     'data_path': './ara.csv',
#     'src_lang': 'English',
#     'tgt_lang': 'Arabic'
# }


# def build_tokenizer(config, data, lang):
#     tokenizer_path = Path(config['tokenizer_path']) / f"tokenizer_{lang}.json"
#     if not tokenizer_path.exists():
#         tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
#         tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#         trainer = trainers.WordLevelTrainer(
#             special_tokens=["<unk>", "<start>", "<end>", "<pad>"], min_frequency=2
#         )
#         tokenizer.train_from_iterator(data[lang], trainer=trainer)
#         tokenizer.save(str(tokenizer_path))  # Convert PosixPath to string
#     else:
#         tokenizer = Tokenizer.from_file(str(tokenizer_path))
#     return tokenizer


# class CreateTrainingDataForTransformer(Dataset):
#     def __init__(self, config, data):
#         self.config = config
#         self.data = data
#         self.src_tokenizer = build_tokenizer(config, data, config['src_lang'])
#         self.tgt_tokenizer = build_tokenizer(config, data, config['tgt_lang'])

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         src_text = self.data[config['src_lang']][idx]
#         tgt_text = self.data[config['tgt_lang']][idx]
#         src_tokenized = self.src_tokenizer.encode(src_text).ids
#         tgt_tokenized = self.tgt_tokenizer.encode(tgt_text).ids
#         return torch.tensor(src_tokenized), torch.tensor(tgt_tokenized)


# def process_data(data):
#     data = data.apply(lambda x: x.astype(str).str.lower())
#     data = data.apply(lambda x: x.astype(str).str.replace(r'[^\w\s]', '', regex=True))
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\d+', '', regex=True))
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\n', '', regex=True))
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\r', '', regex=True))
#     data = data.apply(lambda x: x.astype(str).str.replace(r'\s+', ' ', regex=True))
#     # data = data.apply(lambda x: str(tb(x).correct()))
#     return data


# def get_max_seq_len(data):
#     max_seq_len = 0
#     for i in range(len(data)):
#         src_text = data[config['src_lang']][i]
#         tgt_text = data[config['tgt_lang']][i]
#         src_tokenized = len(src_text.split())
#         tgt_tokenized = len(tgt_text.split())
#         if src_tokenized > max_seq_len:
#             max_seq_len = src_tokenized
#         if tgt_tokenized > max_seq_len:
#             max_seq_len = tgt_tokenized
#     return max_seq_len


# def train_transformer(config, data):
#     data = process_data(data)
#     training_data = CreateTrainingDataForTransformer(config, data)
#     training_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

#     src_vocab_size = len(training_data.src_tokenizer.get_vocab())
#     tgt_vocab_size = len(training_data.tgt_tokenizer.get_vocab())

#     model = build_transformer(
#         src_seq_len=get_max_seq_len(data[config['src_lang']]),
#         tgt_seq_len=get_max_seq_len(data[config['tgt_lang']]),
#         src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
#         d_model=512, h=8, N=6, dropout=0.1, d_ff=2048
#     )
#     model.train()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     criterion = torch.nn.CrossEntropyLoss(ignore_index=training_data.tgt_tokenizer.token_to_id("<pad>"))

#     for epoch in range(10):
#         for src_tokenized, tgt_tokenized in training_data_loader:
#             optimizer.zero_grad()
#             tgt_input = tgt_tokenized[:, :-1]
#             tgt_output = tgt_tokenized[:, 1:]
#             output = model(src_tokenized, tgt_input)
#             loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
#             loss.backward()
#             optimizer.step()
#             print(f'Epoch: {epoch}, Loss: {loss.item()}')


# if __name__ == '__main__':
#     Path(config['tokenizer_path']).mkdir(parents=True, exist_ok=True)
#     Path(config['model_path']).mkdir(parents=True, exist_ok=True)
#     Path(config['data_path']).mkdir(parents=True, exist_ok=True)

#     data = pd.read_csv('ara.txt', sep='\t', header=None, names=['English', 'Arabic', 'decryption'])
#     data.drop('decryption', axis=1, inplace=True)
#     train_transformer(config, data)
