import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import numpy as np
from create_embeddings import get_tfidf_embedding, get_bag_of_words_embedding, get_glove_embedding
import matplotlib.pyplot as plt


def return_to_tensor(txt):
    return eval("torch." + txt).numpy()


imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')
imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(return_to_tensor)
imdb_df.drop('encode_glove', axis=1, inplace=True)

X = imdb_df['review']
y = imdb_df['sentiment']
X_bert = np.vstack(imdb_df['cls_bert'])
X_glove = get_glove_embedding(imdb_df)

labs = [1 if label == "positive" else 0 for label in y]
labels = torch.tensor(labs).float().unsqueeze(1)

X_dev, X_test, y_dev, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_dev_tfidf, X_test_tfidf = get_tfidf_embedding(X_dev, X_test)
X_dev_bow, X_test_bow = get_bag_of_words_embedding(X_dev, X_test)

X_dev_bert, X_test_bert, y_dev_bert, y_test_bert = train_test_split(X_bert, labels, test_size=0.2, random_state=42)

X_dev_glove, X_test_glove, y_dev_glove, y_test_glove = train_test_split(X_glove, labels, test_size=0.2, random_state=42)

# Grid search code
# This takes a long time to run (overnight), so can comment out grid search to just see the final model performances
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['lbfgs', 'liblinear', 'saga']
}

# tfidf logistic regression
regr_tfidf = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(estimator=regr_tfidf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_dev_tfidf, y_dev)
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# bag of words logistic regression
regr_bow = LogisticRegression(max_iter=1000)
grid_search_bow = GridSearchCV(estimator=regr_bow, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_bow.fit(X_dev_bow, y_dev)
print("Best Hyperparameters:", grid_search_bow.best_params_)
print("Best Score:", grid_search_bow.best_score_)

# bert logistic regression
regr_bert = LogisticRegression(max_iter=1000)
grid_search_bert = GridSearchCV(estimator=regr_bert, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_bert.fit(X_dev_bert, y_dev)
print("Best Hyperparameters:", grid_search_bert.best_params_)
print("Best Score:", grid_search_bert.best_score_)

# glove logistic regression
regr_glove = LogisticRegression(max_iter=1000)
grid_search_glove = GridSearchCV(estimator=regr_glove, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_glove.fit(X_dev_glove, y_dev)
print("Best Hyperparameters:", grid_search_glove.best_params_)
print("Best Score:", grid_search_glove.best_score_)

# Training model w/ final params determined from above grid search
# tfidf logistic regression
regr_tfidf = LogisticRegression(C=10, penalty='l2', solver='liblinear')
regr_tfidf.fit(X_dev_tfidf, y_dev)
y_pred = regr_tfidf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Tfidf accuracy on test: ", accuracy)
print("Tfidf classification report: ")
print(classification_report(y_test, y_pred))

y_pred_proba_tfidf = regr_tfidf.predict_proba(X_test_tfidf)[:, 1]
auc_tfidf = roc_auc_score(y_test, y_pred_proba_tfidf)

# bag of words logistic regression
regr_bow = LogisticRegression(C=0.1, penalty='l2')
regr_bow.fit(X_dev_bow, y_dev)
y_pred = regr_bow.predict(X_test_bow)
accuracy = accuracy_score(y_test, y_pred)
print("Bow accuracy on test: ", accuracy)
print("Bow classification report: ")
print(classification_report(y_test, y_pred))

y_pred_proba_bow = regr_bow.predict_proba(X_test_bow)[:, 1]
auc_bow = roc_auc_score(y_test, y_pred_proba_bow)

# bert logistic regression
regr_bert = LogisticRegression(C=0.1, penalty='l2')
regr_bert.fit(X_dev_bert, y_dev_bert)
y_pred = regr_bert.predict(X_test_bert)
accuracy = accuracy_score(y_test_bert, y_pred)
print("Bert accuracy on test: ", accuracy)
print("Bert classification report: ")
print(classification_report(y_test, y_pred))

y_pred_proba_bert = regr_bert.predict_proba(X_test_bert)[:, 1]
auc_bert = roc_auc_score(y_test, y_pred_proba_bert)

# glove logistic regression
regr_glove = LogisticRegression(C=0.1, penalty='l2')
regr_glove.fit(X_dev_glove, y_dev_glove)
y_pred = regr_glove.predict(X_test_glove)
accuracy = accuracy_score(y_test_glove, y_pred)
print("Glove accuracy on test: ", accuracy)
print("Glove classification report: ")
print(classification_report(y_test, y_pred))

y_pred_proba_glove = regr_glove.predict_proba(X_test_glove)[:, 1]
auc_glove = roc_auc_score(y_test, y_pred_proba_glove)

# ROC Curves
plt.figure(figsize=(10, 8))

# tfidf
fpr_tfidf, tpr_tfidf, _ = roc_curve(y_test, y_pred_proba_tfidf)
plt.plot(fpr_tfidf, tpr_tfidf, label=f'TFIDF (AUC = {auc_tfidf:.2f})')

# Bow
fpr_bow, tpr_bow, _ = roc_curve(y_test, y_pred_proba_bow)
plt.plot(fpr_bow, tpr_bow, label=f'BOW (AUC = {auc_bow:.2f})')

# Bert
fpr_bert, tpr_bert, _ = roc_curve(y_test, y_pred_proba_bert)
plt.plot(fpr_bert, tpr_bert, label=f'BERT (AUC = {auc_bert:.2f})')

# Glove
fpr_glove, tpr_glove, _ = roc_curve(y_test, y_pred_proba_glove)
plt.plot(fpr_glove, tpr_glove, label=f'GLOVE (AUC = {auc_glove:.2f})')


plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Embeddings for Logistic Regression Models')
plt.legend(loc='lower right')
plt.grid()
plt.show()