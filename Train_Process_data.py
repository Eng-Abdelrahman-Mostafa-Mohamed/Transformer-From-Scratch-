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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# import json
# from json import

config = {
    "tokenizer_path": ".",
    "model_path": "traiend_model/best_model_weights.pth",
    "data_path": "./ara.csv",
    "src_lang": "ar",
    "tgt_lang": "en",
    "batch_size": 32,
    "seq_len": 6670,
    "d_model": 512,
    "N": 6,
    "h": 8,
    "dropout": 0.1,
    "d_ff": 2048,
}


def get_all_sents(ds, lang):
    for itm in ds:
        yield itm["translation"][lang]


def build_tokenizer(config, data, lang):
    tokenizer_path = Path(config["tokenizer_path"]) / f"tokenizer_{lang}.json"
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(
            models.WordLevel(unk_token="[<unk>]")
        )  # replace any word not in the vocabulary with <unk>
        tokenizer.pre_tokenizer = (
            pre_tokenizers.Whitespace()
        )  # split the text into words using white spaces
        trainer = trainers.WordLevelTrainer(
            special_tokens=["[<unk>]", "[<start>]", "[<end>]", "[<pad>]"],
            min_frequency=2,  # <start> and <end> tokens are used to indicate the beginning and end of a sentence respectively and <pad> is used to pad the input sequences to the same length during training
        )
        tokenizer.train_from_iterator(get_all_sents(data, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset_from_hugging_face(config):
    train_data = load_dataset(
        "Helsinki-NLP/opus-100",
        f"{config['src_lang']}-{config['tgt_lang']}",
        split="train",
    )
    test_data = load_dataset(
        "Helsinki-NLP/opus-100",
        f"{config['src_lang']}-{config['tgt_lang']}",
        split="test",
    )
    print(f"{train_data[1]['translation'][config['src_lang']]}")
    validation_data = load_dataset(
        "Helsinki-NLP/opus-100",
        f"{config['src_lang']}-{config['tgt_lang']}",
        split="validation",
    )

    train_src_tokenizer = build_tokenizer(config, train_data, config["src_lang"])
    train_tgt_tokenizer = build_tokenizer(config, train_data, config["tgt_lang"])

    ready_train_data = CreateTrainingDataForTransformer(
        train_data,
        train_src_tokenizer,
        train_tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )
    ready_val_data = CreateTrainingDataForTransformer(
        validation_data,
        train_src_tokenizer,
        train_tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_len"],
    )

    max_src_seq_len = 0
    max_tgt_seq_len = 0
    for item in train_data:
        max_src_seq_len = max(
            max_src_seq_len,
            len(
                train_src_tokenizer.encode(item["translation"][config["src_lang"]]).ids
            ),
        )
        max_tgt_seq_len = max(
            max_tgt_seq_len,
            len(
                train_src_tokenizer.encode(item["translation"][config["src_lang"]]).ids
            ),
        )

    print(f"max_src_seq_len {max_src_seq_len} max_tgt_seq_len {max_tgt_seq_len}")

    train_data_loader = DataLoader(
        ready_train_data, batch_size=config["batch_size"], shuffle=True
    )
    val_data_loader = DataLoader(ready_val_data, batch_size=1, shuffle=True)

    return train_data_loader, val_data_loader, train_src_tokenizer, train_tgt_tokenizer


class CreateTrainingDataForTransformer(Dataset):
    def __init__(
        self, data, tokenizer_src, tokenizer_tgt, src_language, target_language, seq_len
    ):
        self.config = config
        self.data = data
        self.seq_len = seq_len
        self.src_tokenizer = tokenizer_src
        self.tgt_tokenizer = tokenizer_tgt

        self.src_language = src_language
        self.target_language = target_language

        self.sos_token = self.src_tokenizer.token_to_id("[<start>]")
        self.eos_token = self.src_tokenizer.token_to_id("[<end>]")
        self.pad_token = self.src_tokenizer.token_to_id("[<pad>]")

        self.src_tokenizer = build_tokenizer(config, data, config["src_lang"])
        self.tgt_tokenizer = build_tokenizer(config, data, config["tgt_lang"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_txt_of_idx = self.data[idx]["translation"][self.src_language]
        tgt_txt_of_idx = self.data[idx]["translation"][self.target_language]

        src_tokens_input_enc = self.src_tokenizer.encode(src_txt_of_idx).ids
        tgt_tokens_input_dec = self.tgt_tokenizer.encode(tgt_txt_of_idx).ids

        src_to_enc_num_padding_needed = self.seq_len - len(src_tokens_input_enc)
        tgt_dec_num_padding_needed = self.seq_len - len(tgt_tokens_input_dec)

        if src_to_enc_num_padding_needed < 0 or tgt_dec_num_padding_needed < 0:
            raise ValueError("The sentense input too long")

        # encoder input i want to make it be like this <sos_token> <src_tokens_input_enc>  <pad_tokens>=<pad_tokens> redundant N=src_to_enc_num_padding_needed

        # انا اسمي محمد
        """ Hint Important note : in training we train on current input tokens to encoder "src lang"

            and internaly we train on prev seq on sec lang using encoder output and previous tokens of decoder input (tgt language ) {THE KEY AND VALUE INPUT TO CROSS ATTENTION BLOCK IN DECODER } 
            
        """

        encoder_input = torch.cat(
            (
                torch.tensor(self.sos_token),
                torch.tensor(src_tokens_input_enc, dtype=torch.int64),
                torch.tensor(self.eos_token, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * src_to_enc_num_padding_needed, dtype=torch.int64
                ),
            ),
            dim=0,
        ).to(device)

        # the input of decoder is using for training the model to predict the target token in training time  itnwill be
        decoder_input = torch.cat(
            (
                torch.tensor(self.sos_token),
                torch.tensor(tgt_tokens_input_dec, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * tgt_dec_num_padding_needed, dtype=torch.int64
                ),
            ),
            dim=0,
        ).to(device)

        lable = torch.cat(
            (
                torch.tensor(self.sos_token),
                torch.tensor(tgt_tokens_input_dec, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * tgt_dec_num_padding_needed, dtype=torch.int64
                ),
                torch.tensor(self.eos_token, dtype=torch.int64),
            ),
            dim=0,
        ).to(device)

        # to check that padding operation applied correctly
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert lable.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # the dimention now is seq length
            "decoder_input": decoder_input,
            "lable": lable,
            # but we want the padded tokens doesn't affect the attention mechanism so we need to create a mask for the encoder and decoder
            # first for batch_size  , second for seq_length and third for the seq_length (we want to reshape mask to work with self attention scores to apply multiblications on  it with the scores) on it
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),
            "decoder_mask": (decoder_input != self.pad_token)
            .type(torch.int64)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(decoder_input.size(0)),
            "src_txt": src_txt_of_idx,
            "tgt_txt": tgt_txt_of_idx,
        }


def causal_mask(tgt_seq_len):
    mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len))
    return mask == 0


def process_data(data):
    data = data.apply(lambda x: x.astype(str).str.lower())
    data = data.apply(lambda x: x.astype(str).str.replace(r"[^\w\s]", "", regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r"\d+", "", regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r"\n", "", regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r"\r", "", regex=True))
    data = data.apply(lambda x: x.astype(str).str.replace(r"\s+", " ", regex=True))
    # data = data.apply(lambda x: str(tb(x).correct()))
    return data


def get_model(config, src_vocab_size, tgt_vocab_size):

    transformer = build_transformer(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
        config["N"],
        config["h"],
        config["dropout"],
        config["d_ff"],
    )
    transformer = transformer.to(device)
    return transformer


def train_transformer():
    train_data, val_data, src_tokenizer, tgt_tokenizer = get_dataset_from_hugging_face(
        config
    )
    src_vocab_size = len(src_tokenizer.get_vocab())
    tgt_vocab_size = len(tgt_tokenizer.get_vocab())
    model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        for batch in train_data:
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]
            lable = batch["lable"]
            encoder_mask = batch["encoder_mask"]
            decoder_mask = batch["decoder_mask"]
            optimizer.zero_grad()
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output, lable)
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_data)
        model.eval()
        total_loss = 0
        for batch in val_data:
            encoder_input = batch["encoder_input"]
            decoder_input = batch["decoder_input"]
            lable = batch["lable"]
            encoder_mask = batch["encoder_mask"]
            decoder_mask = batch["decoder_mask"]
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            loss = criterion(output, lable)
            total_loss += loss.item()
        avg_val_loss = total_loss / len
        print(f"Epoch: {epoch} Loss: {total_loss/len(val_data)}")
        if avg_val_loss < avg_train_loss:
            torch.save(model.state_dict(), config["model_path"])
    return model


def load_model(config, src_vocab_size, tgt_vocab_size, model_path):
    model = get_model(config, src_vocab_size, tgt_vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def translate_sentence(model, sentence, src_tokenizer, tgt_tokenizer, max_seq_len):
    model.eval()

    # Tokenize the input sentence
    tokens = src_tokenizer.encode(sentence).ids
    tokens = (
        [src_tokenizer.token_to_id("[<start>]")]
        + tokens
        + [src_tokenizer.token_to_id("[<end>]")]
    )
    tokens += [src_tokenizer.token_to_id("[<pad>]")] * (max_seq_len - len(tokens))
    tokens = torch.tensor(tokens).unsqueeze(0)

    # Create the input mask
    src_mask = (
        (tokens != src_tokenizer.token_to_id("[<pad>]")).unsqueeze(1).unsqueeze(2).int()
    )

    # Generate predictions
    output = model.generate(tokens, max_length=max_seq_len, src_mask=src_mask)

    # Decode the output
    output_tokens = output.squeeze().tolist()
    translated_sentence = tgt_tokenizer.decode(output_tokens, skip_special_tokens=True)

    return translated_sentence


if __name__ == "__main__":

    model = train_transformer()

    src_tokenizer = build_tokenizer(config, data=None, lang=config["src_lang"])
    tgt_tokenizer = build_tokenizer(config, data=None, lang=config["tgt_lang"])

    # Load model
    model = load_model(
        config,
        len(src_tokenizer.get_vocab()),
        len(tgt_tokenizer.get_vocab()),
        config["model_path"],
    )

    # Translate a sentence
    sentence = "أهلاً بك في عالم الذكاء الاصطناعي"
    translated_sentence = translate_sentence(
        model, sentence, src_tokenizer, tgt_tokenizer, config["seq_len"]
    )
    print(translated_sentence)
