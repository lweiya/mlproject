import os
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# 设置随机种子
def set_seed(n=42):
    import random
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)

# bertEncoder模型
class BertEncoder(nn.Module):

    def __init__(self, bert, tokenizer, pooling='first-last-avg', mask_pooling=True):
        super().__init__()
        self.pooling = pooling
        self.mask_pooling = mask_pooling
        if mask_pooling:
            self.avg_func = self.masked_mean
        else:
            self.avg_func = self.mean

        self.bert = bert
        self.tokenizer = tokenizer

        self.bert.config.output_hidden_states = True

    def forward(self, x):
        out = self.bert(**x)
        if self.pooling == 'cls':
            sentence_embedding = out.last_hidden_state[:, 0]
        elif self.pooling == 'pooler':
            sentence_embedding = out.pooler_output
            sentence_embedding = torch.tanh(sentence_embedding)
        elif self.pooling == 'last-avg':
            state = out.last_hidden_state
            mask = x['attention_mask']
            sentence_embedding = self.avg_func(state, mask)
        elif self.pooling == 'first-last-avg':
            mask = x['attention_mask']
            first_state = out.hidden_states[0]
            last_state = out.hidden_states[-1]
            first_embedding = self.avg_func(first_state, mask)
            last_embedding = self.avg_func(last_state, mask)
            sentence_embedding = (first_embedding + last_embedding) / 2
        return sentence_embedding

    def masked_mean(self, state, mask):
        mask_expaned = mask.unsqueeze(-1).expand(state.size()).float()
        sum_state = torch.sum(state * mask_expaned, 1)
        sum_mask = mask_expaned.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_state / sum_mask

    def mean(self, state, mask):
        return torch.mean(state, 1)

    def tokenize(self, texts, max_length=64):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def encode(self, texts, batch_size=64, max_length=128, show_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        embedings = []
        with torch.no_grad():
            bar = range(0, len(texts), batch_size)
            if show_bar:
                bar = tqdm(bar)
            for start in bar:
                end = min(start+batch_size, len(texts))
                encodings = self.tokenize(texts[start: end], max_length)
                encodings = {k: v.to(self.bert.device) for k, v in encodings.items()}
                embedding = self.forward(encodings)
                embedings.append(embedding.cpu())
        embedings = torch.cat(embedings, 0)
        return embedings

    def save(self, path):
        self.bert.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, 'bert_encoder_config.json'), 'w') as f_out:
            config = {
                'pooling': self.pooling,
                'mask_pooling': self.mask_pooling
            }
            json.dump(config, f_out, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path):
        bert = AutoModel.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        with open(os.path.join(path, 'bert_encoder_config.json'), 'r') as f_in:
            config = json.load(f_in)
        return BertEncoder(
            bert=bert,
            tokenizer=tokenizer,
            pooling=config['pooling'],
            mask_pooling=config['mask_pooling']
        )

    def compute_loss(self, x, y, temperature=0.05):
        device = x.device
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        scores = torch.mm(norm_x, norm_y.transpose(0, 1)) * temperature
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
        loss = F.cross_entropy(scores, labels)
        return loss

    def fit(self, params, train_df, val_df=None):
        pass

# SimCSE模型
class SimCSE(BertEncoder):
    def __init__(self, model_path, dropout, pooling, max_length):
        config = AutoConfig.from_pretrained(model_path)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        bert = AutoModel.from_pretrained(model_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        super().__init__(
            bert=bert,
            tokenizer=tokenizer,
            pooling=pooling
        )

        self.max_length = max_length

    # 计算损失
    def compute_loss(self, y_pred, temperature=0.05):
        device = y_pred.device
        # 文本的数量
        num_texts = y_pred.shape[0]
        label = torch.arange(num_texts).to(device)
        label = (label + int(num_texts / 2)) % num_texts
        similarity = F.cosine_similarity(
            y_pred.unsqueeze(1),
            y_pred.unsqueeze(0),
            dim=-1
        )
        mask = torch.eye(num_texts).to(device) * 1e12
        similarity = similarity - mask
        similarity = similarity / temperature
        loss = F.cross_entropy(similarity, label)
        return loss

# 训练数据集
class TrainDataset(Dataset):
    def __init__(self, texts) -> None:
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


# 打印模型训练模板
def get_desc(epoch, loss):
    return '[EPOCH: %d loss: %.6f]' % (epoch, loss)


# 训练模型
def train(base, train_dataset, max_length, epoch, batch_size, lr, pooling, warmup_rate, dropout, output, device):
    # 设置随机种子
    set_seed(42)
    # 设置dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: model.tokenize(x*2, max_length)
    )
    # 设置模型
    model = SimCSE(base, dropout, pooling, max_length).to(device)
    # 设置训练步数
    total_steps = len(train_dataloader) * epoch
    # 设置学习率衰减
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=math.ceil(warmup_rate*total_steps),
        num_training_steps=total_steps
    )
    
    # 设置loss
    l_loss = []
    for e in range(epoch):
        model.train()
        s_loss = 0
        bar = tqdm(train_dataloader, desc=get_desc(e, 0), leave=False)
        for x in bar:
            x = {k: v.to(device) for k, v in x.items()}
            y = model(x)
            loss = model.compute_loss(y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            s_loss += loss.item()
            bar.set_description(get_desc(e, loss.item() / y.shape[0]))

        # 保存模型
        model.save(os.path.join(output, str(e)))
        s_loss = s_loss / len(train_dataset)
        print('Epoch: %d, train_loss: %.6f' % (e, s_loss))
        l_loss.append(s_loss)

    # 保存结果
    json.dump(l_loss, open(os.path.join(output, 'loss.json'), 'w'), ensure_ascii=False, indent=2)
    return model


if __name__ == '__main__':
    # 参数设置
    # DEVICE = 'cpu'
    DEVICE = 'cuda'
    BASE = 'hfl/chinese-roberta-wwm-ext'
    MAX_LENGTH = 80
    EPOCH = 2
    BATCH_SIZE = 64
    LR = 1e-5
    WARMUP_RATE = 0.1
    DROPOUT = 0.1
    INPUT = '../data/unlabeled_train_data.txt'
    OUTPUT = '../training/pretrained_model'

    # 数据集
    raw_data = open(INPUT).readlines()
    # 去掉换行符
    raw_data = [text.strip() for text in raw_data]
    # 去掉空白
    raw_data = [text for text in raw_data if text]
    train_dataset = TrainDataset(raw_data)
 
    # 训练
    train(
        base=BASE,
        train_dataset=train_dataset,
        max_length=MAX_LENGTH,
        epoch=EPOCH,
        batch_size=BATCH_SIZE,
        lr=LR,
        pooling='first-last-avg',
        warmup_rate=WARMUP_RATE,
        dropout=DROPOUT,
        output=OUTPUT,
        device=DEVICE
    )
