import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from Transformer import Transformer, build_transformer

# Read and preprocess the data
data = pd.read_csv('ara.txt', sep='\t', header=None, names=['English', 'Arabic', 'decryption'])
data.drop('decryption', axis=1, inplace=True)
special_words = ['<start>', '<end>', '<pad>', '<unk>']
data = data[~data['English'].isin(special_words)]
data = data[~data['Arabic'].isin(special_words)]

def process_data(data):
    data = data.apply(lambda x: x.astype(str).str.lower())
    data = data.apply(lambda x: x.astype(str).str.replace(r'[^\w\s]', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\d+', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\n', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\r', '', regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r'\s+', ' ', regex=True))
    return data

processed_data = process_data(data)
processed_data['English'] = processed_data['English'].apply(lambda x: '<start> ' + x + ' <end>')
processed_data['Arabic'] = processed_data['Arabic'].apply(lambda x: '<start> ' + x + ' <end>')

# Train the tokenizer
tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(special_tokens=["<unk>", "<start>", "<end>", "<pad>"], min_frequency=2)
tokenizer.train_from_iterator(processed_data['English'].tolist() + processed_data['Arabic'].tolist(), trainer)
tokenizer.post_processor = processors.TemplateProcessing(
    single="<start> $A <end>",
    special_tokens=[
        ("<start>", 1),
        ("<end>", 2),
    ],
)

def tokenize_sequences(data, column):
    return data[column].apply(lambda x: tokenizer.encode(x).ids)

processed_data['English_tokens'] = tokenize_sequences(processed_data, 'English')
processed_data['Arabic_tokens'] = tokenize_sequences(processed_data, 'Arabic')

# Create a dataset and dataloader
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data.iloc[idx]['English_tokens']
        tgt = self.data.iloc[idx]['Arabic_tokens']
        return torch.tensor(src), torch.tensor(tgt)

dataset = TranslationDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the training loop
def train_transformer(model, dataloader, epochs, optimizer, criterion):
    model.train()
    for epoch in range(epochs):
        for src, tgt in dataloader:
            src_mask = (src != tokenizer.token_to_id("<pad>")).unsqueeze(-2)
            tgt_mask = (tgt != tokenizer.token_to_id("<pad>")).unsqueeze(-2)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Initialize the model, optimizer, and loss function
src_vocab_size = len(tokenizer.get_vocab())
tgt_vocab_size = len(tokenizer.get_vocab())
src_seq_len = max(processed_data['English_tokens'].apply(len))
tgt_seq_len = max(processed_data['Arabic_tokens'].apply(len))
d_model = 512
N = 6
h = 8
dropout = 0.1
d_ff = 2048

model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
epochs = 10
# Train the model
train_transformer(model, dataloader,optimizer,epochs ,criterion)