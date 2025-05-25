# rainbow_yu exp7.dataloader 🐋✨
# 数据预处理代码

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from pathlib import Path
from collections import Counter

# -----------------------
# 🔧 本地数据加载部分
# -----------------------
def read_local_imdb(split_dir, max_per_class=700):
    data = []
    for label_dir in ['pos', 'neg']:
        label_path = Path(split_dir) / label_dir
        print(f"📁 读取子目录: {label_path}")
        count = 0
        for file_path in label_path.glob('*.txt'):
            if count >= max_per_class:
                break
            if count % 100 == 0:
                print(f"📝 正在读取第 {count} 个文件: {file_path.name}")
            try:
                with open(file_path, encoding='utf8') as f:
                    text = f.read().strip()
                    label = 'pos' if label_dir == 'pos' else 'neg'
                    data.append((label, text))
                    count += 1
            except Exception as e:
                print(f"❌ 读取失败: {file_path}，错误: {e}")
    return data



# 替代 torchtext.datasets.IMDB
train_iter = read_local_imdb('../../datasets/imdb/train',700)
test_iter = read_local_imdb('../../datasets/imdb/test',300)

# -----------------------
# ✅ 原有函数保持不变
# -----------------------

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

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
