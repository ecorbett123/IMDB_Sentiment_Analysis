import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from create_embeddings import get_tfidf_embedding, get_bag_of_words_embedding


def return_to_tensor(txt):
    return eval("torch." + txt).numpy()


imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')
imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(return_to_tensor)

X = imdb_df['review']
y = imdb_df['sentiment']
X_bert = imdb_df['cls_bert']

labs = [1 if label == "positive" else 0 for label in y]
#embeddings = get_batch_embeddings(X, tokenizer, model, device, batch_size=32)
labels = torch.tensor(labs).float().unsqueeze(1)

X_dev, X_test, y_dev, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_dev_tfidf, X_test_tfidf = get_tfidf_embedding(X_dev, X_test)
X_dev_bow, X_test_bow = get_bag_of_words_embedding(X_dev, X_test)

X_dev_bert, X_test_bert, y_dev_bert, y_test_bert = train_test_split(X_bert, labels, test_size=0.2, random_state=42)

# tfidf logistic regression
# regr_tfidf = LogisticRegression()
# regr_tfidf.fit(X_dev_tfidf, y_dev)
# y_pred = regr_tfidf.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)
#
# # bag of words logistic regression
# regr_bow = LogisticRegression()
# regr_bow.fit(X_dev_bow, y_dev)
# y_pred = regr_bow.predict(X_test_bow)
# accuracy = accuracy_score(y_test, y_pred)
# print(accuracy)

# bert logistic regression
regr_bert = LogisticRegression()
regr_bert.fit(X_dev_bert, y_dev_bert)
y_pred = regr_bert.predict(X_test_bert)
accuracy = accuracy_score(y_test_bert, y_pred)
print(accuracy)

# glove logistic regression
