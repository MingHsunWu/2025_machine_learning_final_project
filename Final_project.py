# -*- coding: utf-8 -*-
"""
Created on Dec 1 2025

@author: Ming Hsun
"""
#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,matthews_corrcoef
from sklearn.utils import resample
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

#%%
'資料讀取和預處理 '
# 讀取資料
train_df = pd.read_csv("./data/training.csv")
valid_df = pd.read_csv("./data/validation.csv")
test_df = pd.read_csv("./data/test.csv")
total_df = pd.concat([train_df, valid_df, test_df])
data_sentences = total_df['text'].values
data_labels = total_df['label'].values

# 轉成詞向量
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = [tokenizer.encode(sentences,add_special_tokens=True,max_length = 100,padding='max_length') for sentences in data_sentences]

# BERT所需mask
attention_masks = []
attention_masks = [[float(k>0) for k in seq] for seq in input_ids]


X_train, X_test, y_train, y_test, train_mask, test_mask = train_test_split(
    input_ids, data_labels, attention_masks, test_size=0.2, random_state=31
)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)



#%%
'建立model'
# 訓練參數
batch_size = 32
epochs = 50
lr = 2e-5

# 建立DataLoader
training_data = TensorDataset(X_train, train_mask, y_train)
training_dataloader = DataLoader(training_data, sampler=RandomSampler(training_data), batch_size=batch_size)

validation_data = TensorDataset(X_test, test_mask, y_test)
validation_dataloader = DataLoader(validation_data, sampler=RandomSampler(validation_data), batch_size=batch_size)

# 定義模型
class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name="bert-base-uncased", num_labels=6,
                 dropout_prob=0.1, hidden_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        bert_hidden_size = self.bert.config.hidden_size  
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_labels),
            nn.Softmax()
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  
        x = self.dropout(pooled)
        output = self.classifier(x)     
        return output
    
model = BertClassifier(pretrained_model_name="bert-base-uncased", num_labels=6).to(device)

# 凍結BERT的權重更新
for param in model.bert.parameters():
    param.requires_grad = False

# 只把 classifier 參數傳給 optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, eps=1e-8)

total_steps = len(training_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
criterion = torch.nn.CrossEntropyLoss()
#%%
'訓練'
train_losses = []
val_losses = []
val_accuracies = []
val_mccs = []
epoch_list = []
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    # training
    for idx, data in enumerate(training_dataloader):
        optimizer.zero_grad()

        data = tuple(t.to(device) for t in data)
        data_inputs_ids, data_inputs_mask, data_labels = data  

        logits = model(data_inputs_ids, attention_mask=data_inputs_mask)  
        loss = criterion(logits, data_labels)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(training_dataloader)
    train_losses.append(avg_train_loss)

    # validation
    model.eval()
    total_eval_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            logits = model(b_input_ids, attention_mask=b_input_mask)
            loss = criterion(logits, b_labels)

            total_eval_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = b_labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_mcc = matthews_corrcoef(all_labels, all_preds)

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    val_mccs.append(val_mcc)
    epoch_list.append(epoch)
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"\tTrain Loss: {avg_train_loss:.4f}")
    print(f"\tValidation Loss: {avg_val_loss:.4f}")
    print(f"\tValidation Accuracy: {val_acc:.4f}")
    #print(f"\tValidation MCC: {val_mcc:.4f}")
    print("------")










