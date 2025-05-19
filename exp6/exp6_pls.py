import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from io import open
import math
import time

# 全局变量定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0  # 句子开始标记
EOS_token = 1  # 句子结束标记
MAX_LENGTH = 10  # 最大句子长度


# 语言处理类
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": SOS_token, "EOS": EOS_token}  # 初始化标记
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # 初始包含SOS和EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 数据预处理函数
def unicode_to_ascii(s):
    """将Unicode转换为ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    """规范化句子：转ASCII、小写、去标点"""
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)  # 在标点前加空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 移除非字母和标点字符
    return s.strip()


def read_langs(lang1, lang2, reverse=False):
    """读取数据文件并分割为句子对"""
    lines = open(f"data/{lang1}-{lang2}.txt", encoding="utf-8").read().strip().split('\n')
    pairs = [[normalize_string(s) for s in line.split('\t')[:2]] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filter_pair(pair):
    """过滤短句子和特定前缀的句子"""
    eng_prefixes = ("i am ", "i'm ", "he is", "he's ", "she is", "she's ",
                    "you are", "you're ", "we are", "we're ", "they are", "they're ")
    return len(pair[0].split(' ')) < MAX_LENGTH and \
        len(pair[1].split(' ')) < MAX_LENGTH and \
        pair[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    """批量过滤句子对"""
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1, lang2, reverse=False):
    """完整数据预处理流程"""
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print(f"读取到 {len(pairs)} 个句子对")
    pairs = filter_pairs(pairs)
    print(f"过滤后剩余 {len(pairs)} 个句子对")

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print(f"源语言词汇量: {input_lang.n_words}, 目标语言词汇量: {output_lang.n_words}")
    return input_lang, output_lang, pairs


# 模型定义：编码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_tensor, hidden_tensor):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden_tensor)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 模型定义：基础解码器（无注意力）
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = F.relu(embedded)
        output, hidden = self.gru(embedded, hidden_tensor)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 模型定义：注意力解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden_tensor, encoder_outputs):
        embedded = self.embedding(input_tensor).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # 计算注意力权重
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_tensor[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 合并上下文向量和嵌入向量
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        # 输入GRU层
        output, hidden = self.gru(output, hidden_tensor)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 训练辅助函数
def indexes_from_sentence(lang, sentence):
    """句子转索引列表"""
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def tensor_from_sentence(lang, sentence):
    """句子转张量"""
    indexes = indexes_from_sentence(lang, sentence)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(pair):
    """句子对转张量对"""
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# 训练函数
teacher_forcing_ratio = 0.5  # 教师强制比例


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.init_hidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    loss = 0.0

    # 编码器前向传播
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # 解码器前向传播
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 教师强制：直接使用目标输出作为输入
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # 下一时刻输入为目标词
    else:
        # 非教师强制：使用模型预测作为输入
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach to avoid gradient tracking
            if decoder_input.item() == EOS_token:
                break  # 遇到EOS提前终止

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# 训练迭代控制函数
def train_iters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0.0
    plot_loss_total = 0.0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # 与LogSoftmax配合使用
    training_pairs = [tensors_from_pair(random.choice(pairs)) for _ in range(n_iters)]

    for iter in range(1, n_iters + 1):
        input_tensor, target_tensor = training_pairs[iter - 1]
        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(f"迭代 {iter}/{n_iters}，耗时 {time_since(start, iter / n_iters)}，平均损失: {print_loss_avg:.4f}")
            print_loss_total = 0

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    show_plot(plot_losses)  # 绘制损失曲线


def time_since(since, percent):
    """计算耗时"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return f"{as_minutes(s)} (- {as_minutes(rs)})"


def as_minutes(s):
    """秒转分钟"""
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {s}s"


def show_plot(points):
    """绘制损失曲线"""
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("output/loss")


# 评估函数
def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        sentence = normalize_string(sentence)
        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

        # 编码器前向传播
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        # 解码器初始化
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        # 解码循环
        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluate_randomly(encoder, decoder, n=10):
    """随机评估"""
    for _ in range(n):
        pair = random.choice(pairs)
        print(f"> {pair[0]}")
        print(f"= {pair[1]}")
        output_words, _ = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words[:-1])  # 去除EOS标记
        print(f"< {output_sentence}\n")


# 注意力可视化函数
def show_attention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    input_words = input_sentence.split(' ') + ['<EOS>']

    # 设置 x/y 轴的 ticks 和 labels
    ax.set_xticks(range(len(input_words)))
    ax.set_yticks(range(len(output_words)))

    ax.set_xticklabels(input_words, rotation=90)
    ax.set_yticklabels(output_words)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.savefig("output/attention")



def evaluate_and_show_attention(input_sentence):
    """评估并可视化注意力"""
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence)
    print(f"输入: {input_sentence}")
    print(f"输出: {' '.join(output_words[:-1])}")  # 去除EOS标记
    show_attention(input_sentence, output_words, attentions)


# ----------------------
# 主程序执行流程
# ----------------------
if __name__ == "__main__":
    # 1. 数据预处理
    input_lang, output_lang, pairs = prepare_data('eng', 'fra', reverse=True)  # 法语->英语，reverse=True

    # 2. 初始化模型（使用带注意力的解码器）
    hidden_size = 256
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    # 3. 训练模型
    print("\n开始训练...")
    train_iters(encoder, attn_decoder, n_iters=750, print_every=50, learning_rate=0.01)

    # 4. 随机评估
    print("\n随机评估结果:")
    evaluate_randomly(encoder, attn_decoder, n=5)

    # 5. 注意力可视化（示例句子）
    print("\n注意力可视化示例:")
    evaluate_and_show_attention("elle a cinq ans de moins que moi.")
    evaluate_and_show_attention("elle est trop petit.")