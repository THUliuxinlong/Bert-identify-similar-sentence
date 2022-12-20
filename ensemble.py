import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report

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
valid_df = pd.read_csv('./data/dev.csv')
valid_df['text'] = '句子1：' + valid_df['query1'].astype(str) + '句子2：' + valid_df['query2'].astype(str)

PRE_TRAINED_MODEL_NAME_1 = 'hfl/chinese-bert-wwm-ext'
tokenizer1 = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME_1)
PRE_TRAINED_MODEL_NAME_2 = 'hfl/chinese-roberta-wwm-ext'
tokenizer2 = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME_2)

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

test_df = valid_df
test_data_loader1 = create_data_loader(test_df, tokenizer1, MAX_LEN, BATCH_SIZE)
test_data_loader2 = create_data_loader(test_df, tokenizer2, MAX_LEN, BATCH_SIZE)

# 定义模型
class IsSimilarClassifier(nn.Module):
    def __init__(self, n_classes, PRE_TRAINED_MODEL_NAME):
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

model1 = IsSimilarClassifier(n_classes=2, PRE_TRAINED_MODEL_NAME=PRE_TRAINED_MODEL_NAME_1)
model1.load_state_dict(torch.load('best_model1_state.pth'))
model1 = model1.to(device)
model2 = IsSimilarClassifier(n_classes=2, PRE_TRAINED_MODEL_NAME=PRE_TRAINED_MODEL_NAME_2)
model2.load_state_dict(torch.load('best_model2_state.pth'))
model2 = model2.to(device)
# print(model)

# 和评估函数类似，但是存储了新闻的文本和预测概率
def get_predictions(model1, model2, data_loader1, data_loader2):
    model1 = model1.eval()
    model2 = model2.eval()

    predictions = []
    real_values = []

    with torch.no_grad():
        for d1, d2 in zip(data_loader1, data_loader2):

            input_ids1 = d1["input_ids"].to(device)
            attention_mask1 = d1["attention_mask"].to(device)
            label = d1["label"].to(device)

            outputs1 = model1(
            input_ids=input_ids1,
            attention_mask=attention_mask1
            )

            input_ids2 = d2["input_ids"].to(device)
            attention_mask2 = d2["attention_mask"].to(device)

            outputs2 = model2(
            input_ids=input_ids2,
            attention_mask=attention_mask2
            )

            outputs = (outputs1 + outputs2) / 2

            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            real_values.extend(label)

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values

y_pred, y_real_label = get_predictions(
    model1,
    model2,
    test_data_loader1,
    test_data_loader2
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