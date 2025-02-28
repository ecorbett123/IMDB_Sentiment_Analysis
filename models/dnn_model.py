
### IMPORT PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from create_embeddings import get_tfidf_embedding, get_bag_of_words_embedding
import torch

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('punkt')

# Read in embedding file, 'review' column already preprocessed - bert column: 'cls_bert', glove column: 'encode_glove'
# Download, split, and tokenize the data for processing
imdb_df = pd.read_csv('../imdb_with_glove_bert_embeddings.csv')

X = imdb_df['review']
y = imdb_df['sentiment']

labs = [1 if label == "positive" else 0 for label in y]
labels = torch.tensor(labs).float().unsqueeze(1)

X_dev, X_test, y_dev, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42)
X_dev_tfidf, X_test_tfidf = get_tfidf_embedding(X_dev, X_test)
X_dev_bow, X_test_bow = get_bag_of_words_embedding(X_dev, X_test)

imdb_df['encode_glove'] = imdb_df['encode_glove'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
)
imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(
    lambda x: np.fromstring(x.strip('tensor([])'), sep=',') if isinstance(x, str) else x.numpy() if isinstance(x, torch.Tensor) else x
)

# Vertically stack GloVe embeddings for train and test sets
X_dev_glove = np.vstack(imdb_df.loc[X_dev.index, 'encode_glove'].values)
X_test_glove = np.vstack(imdb_df.loc[X_test.index, 'encode_glove'].values)

# Vertically stack BERT embeddings for train and test sets
X_dev_bert = np.vstack(imdb_df.loc[X_dev.index, 'cls_bert'].values)
X_test_bert = np.vstack(imdb_df.loc[X_test.index, 'cls_bert'].values)

imdb_df.head()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import tensorflow as tf
from sklearn.metrics import classification_report
from keras_tuner import RandomSearch
from sklearn.metrics import roc_curve, auc

"""## TF-IDF"""

# tfidf
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_dev_tfidf.shape[1])) # try 128/64
model.add(Dense(units=32, activation='relu')) # try 64/32
model.add(Dropout(0.5)) #try 0.5/0.3
model.add(Dense(units=16, activation='relu')) # try 16/8
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):  # Force the use of GPU
    history_tfidf = model.fit(
        X_dev_tfidf, y_dev,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        verbose=1)

loss, accuracy = model.evaluate(X_test_tfidf, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = model.predict(X_test_tfidf)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Hyperparameter Tuning
def build_tfidf_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Choice('units_input', [64, 128]),
                    activation='relu', input_dim=X_dev_tfidf.shape[1]))
    model.add(Dense(units=hp.Choice('units_hidden', [32, 64]),
                    activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout', [0.2, 0.3])))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_tfidf = RandomSearch(
    build_tfidf_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='tfidf_quick_tune'
)

with tf.device('/GPU:0'):
    tuner_tfidf.search(
        X_dev_tfidf.toarray(), y_dev,
        validation_split=0.2,
        epochs=5,  # Use only 5 epochs
        batch_size=32,
        verbose=1
    )

best_hps_tfidf = tuner_tfidf.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the input layer is {best_hps_tfidf.get('units_input')}.
The optimal learning rate is {best_hps_tfidf.get('learning_rate')}.
The optimal dropout rate is {best_hps_tfidf.get('dropout')}.
""")

# Build the best model
best_model_tfidf = tuner_tfidf.hypermodel.build(best_hps_tfidf)

# Train the best model
with tf.device('/GPU:0'):
    history_tfidf = best_model_tfidf.fit(
        X_dev_tfidf.toarray(), y_dev,  # Convert sparse matrix to dense
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )

y_pred = best_model_tfidf.predict(X_test_tfidf)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

"""## BOW"""

# bow
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_dev_bow.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_bow = model.fit(X_dev_bow, y_dev,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.2,
                      verbose=1)

loss, accuracy = model.evaluate(X_test_bow, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = model.predict(X_test_bow)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Hyperparameter Tuning - Tune manually above with the session crashed same as tf-idf
def build_bow_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Choice('units_input', [64, 128]),
                    activation='relu', input_dim=X_dev_tfidf.shape[1]))
    model.add(Dense(units=hp.Choice('units_hidden', [32, 64]),
                    activation='relu'))
    model.add(Dropout(rate=hp.Choice('dropout', [0.2, 0.3])))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner_bow = RandomSearch(
    build_tfidf_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='my_dir',
    project_name='tfidf_quick_tune'
)

with tf.device('/GPU:0'):
    tuner_bow.search(
        X_dev_bow.toarray(), y_dev,
        validation_split=0.2,
        epochs=5,  # Use only 5 epochs
        batch_size=32,
        verbose=1
    )

best_hps_bow = tuner_bow.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the input layer is {best_hps_bow.get('units_input')}.
The optimal learning rate is {best_hps_bow.get('learning_rate')}.
The optimal dropout rate is {best_hps_bow.get('dropout')}.
""")

# Build the best model
best_model_bow = tuner_bow.hypermodel.build(best_hps_tfidf)

# Train the best model
with tf.device('/GPU:0'):
    history_tfidf = best_model_bow.fit(
        X_dev_tfidf.toarray(), y_dev,  # Convert sparse matrix to dense
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )

y_pred = best_model_bow.predict(X_test_bow)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

"""## Glove Embedding"""

# glove
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_dev_glove.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):
  history_glove = model.fit(X_dev_glove, y_dev,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.2,
                      verbose=1)

loss, accuracy = model.evaluate(X_test_glove, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Hyperparameter Tuning
def build_model(hp):
    model = Sequential()
    # Input layer with tunable units
    model.add(Dense(units=hp.Int('units_input', min_value=32, max_value=128, step=32),
                    activation='relu', input_dim=X_dev_glove.shape[1]))
    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):  # Tune the number of layers
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=16, max_value=64, step=16),
                        activation=hp.Choice('activation', ['relu', 'tanh'])))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile with tunable learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # Limit to 20 trials
    executions_per_trial=1,
    directory='my_dir',
    project_name='random_search'
)

tuner.search(X_dev_glove, y_dev,
                 validation_split=0.2,
                 epochs=10,
                 batch_size=32,
                 verbose=1)

best_hps_glove = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the input layer is {best_hps_glove.get('units_input')}.
The optimal learning rate is {best_hps_glove.get('learning_rate')}.
The optimal dropout rate is {best_hps_glove.get('dropout')}.
""")

# Build the best model using the best hyperparameters for GloVe
best_model_glove = tuner.hypermodel.build(best_hps_glove)

# Train the best model for GloVe
with tf.device('/GPU:0'):
    history_glove = best_model_glove.fit(
        X_dev_glove, y_dev,  # Use GloVe embeddings here
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1
    )

loss, accuracy = best_model_glove.evaluate(X_test_glove, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = best_model_glove.predict(X_test_glove)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

roc_auc

"""## Bert Embedding"""

# bert
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X_dev_bert.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

with tf.device('/GPU:0'):
  history_bert = model.fit(X_dev_bert, y_dev,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.2,
                      verbose=1)

loss, accuracy = model.evaluate(X_test_bert, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Hyperparameter Tuning
def build_bert_model(hp):
    model = Sequential()
    # Input layer with tunable units
    model.add(Dense(units=hp.Int('units_input', min_value=64, max_value=256, step=32),
                    activation='relu', input_dim=X_dev_bert.shape[1]))
    # Hidden layers
    for i in range(hp.Int('num_layers', 1, 3)):  # Tune the number of layers
        model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                        activation=hp.Choice('activation', ['relu', 'tanh'])))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile with tunable learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(
                        hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_bert_model,
    objective='val_accuracy',
    max_trials=20,  # Limit to 20 random trials
    executions_per_trial=1,  # Number of times to execute each trial
    directory='my_dir',
    project_name='bert_random_search'
)

with tf.device('/GPU:0'):
    tuner.search(
        X_dev_bert, y_dev,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1)

best_hps_bert = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the input layer is {best_hps_bert.get('units_input')}.
The optimal learning rate is {best_hps_bert.get('learning_rate')}.
The optimal dropout rate is {best_hps_bert.get('dropout')}.
""")

# Build the best model using the best hyperparameters
best_model_bert = tuner.hypermodel.build(best_hps_bert)

# Train the best model
with tf.device('/GPU:0'):
    history_bert = best_model_bert.fit(
        X_dev_bert, y_dev,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        verbose=1)

loss, accuracy = best_model_bert.evaluate(X_test_bert, y_test, verbose=0)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

y_pred = best_model_bert.predict(X_test_bert)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
roc_auc

"""## Graph"""

def plot_loss(histories, embedding_names):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'purple']  # Colors for embeddings
    for idx, (history, name) in enumerate(zip(histories, embedding_names)):
        plt.plot(history.history['loss'], label=f'{name} - Training Loss', color=colors[idx], linestyle='-')
        plt.plot(history.history['val_loss'], label=f'{name} - Validation Loss', color=colors[idx], linestyle='--')
    plt.title('Training and Validation Loss for Different Embeddings')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend to the right
    plt.grid(True)
    plt.show()

def plot_accuracy(histories, embedding_names):
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'purple']  # Colors for embeddings
    for idx, (history, name) in enumerate(zip(histories, embedding_names)):
        plt.plot(history.history['accuracy'], label=f'{name} - Training Accuracy', color=colors[idx], linestyle='-')
        plt.plot(history.history['val_accuracy'], label=f'{name} - Validation Accuracy', color=colors[idx], linestyle='--')
    plt.title('Training and Validation Accuracy for Different Embeddings')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend to the right
    plt.grid(True)
    plt.show()

histories = [history_tfidf, history_bow, history_glove, history_bert]
embedding_names = ['TFIDF', 'Bag-of-Words', 'GloVe', 'BERT']

# Plot Loss
plot_loss(histories, embedding_names)

# Plot Accuracy
plot_accuracy(histories, embedding_names)