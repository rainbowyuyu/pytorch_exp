# rainbow_yu exp7.dataloader 🐋✨
# 数据预处理代码

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

tokenizer = get_tokenizer("basic_english")
train_iter = IMDB(split='train')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

train_iter, test_iter = IMDB(split='train'), IMDB(split='test')

def encode(text):
    return vocab(tokenizer(text))

def collate_batch(batch):
    text_list, label_list = [], []
    for label, text in batch:
        text_tensor = torch.tensor(encode(text), dtype=torch.long)
        text_list.append(text_tensor)
        label_list.append(1 if label == 'pos' else 0)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab['<pad>'])
    return text_list, torch.tensor(label_list)
