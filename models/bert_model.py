# -*- coding: utf-8 -*-
### IMPORT PACKAGES
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from transformers import BertTokenizer
import warnings

# Train Bert Model to classify IMDB Comments

nltk.download('punkt_tab')
warnings.filterwarnings(action='ignore')

# Download, split, and tokenize the data for processing
imdb_df = pd.read_csv('../imdb_with_glove_bert_embeddings.csv')

X = imdb_df['review']
y = imdb_df['sentiment']

labs = [1 if label == "positive" else 0 for label in y]
labels = torch.tensor(labs).float().unsqueeze(1)

X_dev, X_test, y_dev, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=512, return_tensors="pt")
val_enc = tokenizer(list(X_val), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Create IMDB dataset from torch dataset
class IMDbDataset(torch.utils.data.Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return {
        'input_ids': self.encodings['input_ids'][idx],
        'attention_mask': self.encodings['attention_mask'][idx],
        'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = IMDbDataset(train_enc, y_train)
val_dataset = IMDbDataset(val_enc, y_val)
test_dataset = IMDbDataset(test_enc, y_test)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate model based on accuracy score
def evaluate():
  model.eval()
  y_preds = []
  y_true = []

  with torch.no_grad():
    for batch in val_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask)
      preds = torch.argmax(outputs.logits, dim=1)
      y_preds.extend(preds.cpu().numpy())
      y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_preds)
    print(f'Validation Accuracy: {accuracy}')
    print(classification_report(y_true, y_preds))

# train model and evaluate performance
def train():
  model.train()
  for e in range(epochs):
    total_loss = 0
    for batch in train_loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)

      optimizer.zero_grad()
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    print(f"Epoch {e + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")
    evaluate()

from transformers import AdamW

## Hyperparameter tuning
epoch_list = [2] #[1,2,3,4,5]
learning_rate_list = [2e-5] #[1e-5, 2e-5, 5e-5, 1e-4]

for epoch in epoch_list:
  for learning_rate in learning_rate_list:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = epoch
    train()

model.eval()
y_preds = []
y_true = []
y_probs = []

with torch.no_grad():
  for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)\

    outputs = model(input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(outputs.logits, dim=1)
    y_preds.extend(preds.cpu().numpy())
    y_true.extend(labels.cpu().numpy())
    y_probs.extend(probs[:, 1].cpu().numpy())

# Print final accuracy scores and AUC
accuracy = accuracy_score(y_true, y_preds)
print(f'Validation Accuracy: {accuracy}')
print(classification_report(y_true, y_preds))

auc = roc_auc_score(y_true, y_probs)
print(f'Validation AUC: {auc}')

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()

model.eval()
y_preds = []
y_true = []
y_probs = []

with torch.no_grad():
  for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)\

    outputs = model(input_ids, attention_mask=attention_mask)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = torch.argmax(outputs.logits, dim=1)
    y_preds.extend(preds.cpu().numpy())
    y_true.extend(labels.cpu().numpy())
    y_probs.extend(probs[:, 1].cpu().numpy())

accuracy = accuracy_score(y_true, y_preds)
print(f'Test Accuracy: {accuracy}')
print(classification_report(y_true, y_preds))

auc = roc_auc_score(y_true, y_probs)
print(f'Test AUC: {auc}')

fpr, tpr, thresholds = roc_curve(y_true, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid()
plt.show()