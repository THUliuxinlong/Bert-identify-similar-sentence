import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from adversarial_training import PGD

import transformers
transformers.logging.set_verbosity_error()


rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
sns.set(context='notebook', style='ticks', rc=rc)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 读取数据
train_df = pd.read_csv('./data/train.csv')
# train_aug = pd.read_csv('./data/back_trans.tsv', names=['id', 'category', 'query1', 'query2', 'label'])
# train_df = train_df.append(train_aug)

valid_df = pd.read_csv('./data/dev.csv')
print(train_df.head())
print(train_df.shape)

# Explore Data Analyze
# 关键词的个数
# category_counts = train_df.groupby("category").label.value_counts()
# print(category_counts)
# sns.countplot(x=train_df['category'])
# plt.show()

# 看正样本和负样本的个数
# label_counts = train_df.groupby("label").label.value_counts()
# print(label_counts)
# sns.countplot(x=train_df['label'])
# plt.show()

# PRE_TRAINED_MODEL_NAME = 'hfl/chinese-bert-wwm-ext'
PRE_TRAINED_MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

# 句长统计
# train_query1_lens = []
# train_query2_lens = []
# valid_query1_lens = []
# valid_query2_lens = []
# for txt in train_df.query1:
#   tokens = tokenizer.encode(txt, max_length=128, truncation=True)
#   train_query1_lens.append(len(tokens))
# for txt in train_df.query2:
#   tokens = tokenizer.encode(txt, max_length=128, truncation=True)
#   train_query2_lens.append(len(tokens))
# for txt in valid_df.query1:
#   tokens = tokenizer.encode(txt, max_length=128, truncation=True)
#   valid_query1_lens.append(len(tokens))
# for txt in valid_df.query2:
#   tokens = tokenizer.encode(txt, max_length=128, truncation=True)
#   valid_query2_lens.append(len(tokens))
#
# grid = plt.GridSpec(2, 2)
# figure = plt.figure()
# ax = figure.add_subplot(grid[0])
# ax.set_title('Train qurey1 length')
# sns.histplot(train_query1_lens, ax=ax)
# ax = figure.add_subplot(grid[1])
# ax.set_title('Train qurey2 length')
# sns.histplot(train_query2_lens, ax=ax)
# ax = figure.add_subplot(grid[2])
# ax.set_title('Vaild qurey1 length')
# sns.histplot(valid_query1_lens, ax=ax)
# ax = figure.add_subplot(grid[3])
# ax.set_title('Vaild qurey2 length')
# sns.histplot(valid_query2_lens, ax=ax)
# plt.tight_layout()
# plt.show()

train_df['text'] = '句子1：' + train_df['query1'].astype(str) + '句子2：' + train_df['query2'].astype(str)
valid_df['text'] = '句子1：' + valid_df['query1'].astype(str) + '句子2：' + valid_df['query2'].astype(str)
print(train_df.head())

class SentenceDataset(Dataset):

    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.label[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SentenceDataset(
        text=df.text.to_numpy(),
        label=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

MAX_LEN = 50
BATCH_SIZE = 32

# valid_df, test_df = train_test_split(valid_df, test_size=0.5, random_state=RANDOM_SEED, shuffle=True)
test_df = valid_df

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(valid_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

# 定义模型
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class IsSimilarClassifier(nn.Module):
    def __init__(self, n_classes):
        super(IsSimilarClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

model = IsSimilarClassifier(n_classes=2)
model = model.to(device)
# print(model)

# 对抗训练
pgd = PGD(model)
K = 3

# 训练
EPOCHS = 10
learning_rate = 2e-5

# AdamW优化器，它纠正了重量衰减，还将使用没有预热步骤的线性调度程序
optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

def CEloss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

# 定义一次训练
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        label = d["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, label)

        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())

        loss.backward(retain_graph=True)

        # # 对抗训练
        # pgd.backup_grad() # 保存正常的grad
        # for t in range(K):
        #     pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        #     if t != K-1:
        #         optimizer.zero_grad()
        #     else:
        #         pgd.restore_grad() # 恢复正常的grad
        #     loss_sum = loss_fn(outputs, label)
        #     loss_sum.backward(retain_graph=True) # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        # pgd.restore() # 恢复embedding参数

        # clip_grad_norm_裁剪模型的梯度来避免梯度爆炸。
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# 评估模型
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, label)

            correct_predictions += torch.sum(preds == label)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# 训练循环并存储训练历史记录
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        CEloss_fn,
        # loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_df)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(valid_df)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    # 转到cpu上，方便后面绘制
    train_acc_cpu = train_acc
    val_acc_cpu = val_acc
    history['train_acc'].append(train_acc_cpu.cpu())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc_cpu.cpu())
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.pth')
        best_accuracy = val_acc


plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.show()

# 测试集评估
test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(test_df)
)
print('test_acc', test_acc.item())

# 和评估函数类似，但是存储了新闻的文本和预测概率
def get_predictions(model, data_loader):
    model = model.eval()

    Sentencetexts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            label = d["label"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            Sentencetexts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return Sentencetexts, predictions, prediction_probs, real_values

y_texts, y_pred, y_pred_probs, y_real_label = get_predictions(
    model,
    test_data_loader
)

class_names = ['dissimilar', 'similar']
print(classification_report(y_real_label, y_pred, target_names=class_names, digits=6))

# 绘制混淆矩阵
def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('Ground truth')
    plt.xlabel('Predicted')
    plt.show()

cm = confusion_matrix(y_real_label, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)