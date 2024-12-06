import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertModel
from transformers import BertTokenizer
from torchtext.vocab import GloVe
import warnings
import nltk
from nltk.corpus import stopwords
import re

nltk.download('punkt_tab')
warnings.filterwarnings(action='ignore')


# CLEANING AND PREPROCESSING
def text_preprocessing(text):
    text = text.lower()

    text = re.sub(r'[,.!?:()"]', '', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    tokens = nltk.word_tokenize(text)
    return ' '.join(tokens)


# Bag of words embedding
def get_bag_of_words_embedding(X_dev, X_test):
    word_embed_bow = CountVectorizer()
    X_dev_bow = word_embed_bow.fit_transform(X_dev)
    X_test_bow = word_embed_bow.transform(X_test)
    return X_dev_bow, X_test_bow

# TFIDF embedding
def get_tfidf_embedding(X_dev, X_test):
    word_embed_tfidf = TfidfVectorizer()
    X_dev_tfidf = word_embed_tfidf.fit_transform(X_dev)
    X_test_tfidf = word_embed_tfidf.transform(X_test)
    return X_dev_tfidf, X_test_tfidf


# GloVe embedding
def glove_sentence_embedding(sentence, max_length, embedding_dim, embeddings):
    words = sentence.split()
    num_words = min(len(words), max_length)
    embedding_sentence = np.zeros((max_length, embedding_dim))

    for i in range(num_words):
        word = words[i]
        if word in embeddings.stoi:
            embedding_sentence[i] = embeddings.vectors[embeddings.stoi[word]]

    return embedding_sentence.flatten()


def get_glove_embedding(imdb_df):
    embeddings = GloVe(name='6B', dim=100)

    # Set the maximum sentence length and embedding dimension
    max_length = 100
    embedding_dim = 100

    imdb_df['encode_glove'] = imdb_df['review'].apply(
        lambda sentence: glove_sentence_embedding(sentence, max_length, embedding_dim, embeddings))
    X_glove = np.vstack(imdb_df['encode_glove'])
    return X_glove


# Bert embedding
def get_bert_embedding(imdb_df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    imdb_df['cls_bert'] = imdb_df['review'].apply(
        lambda sentence: get_cls_sentence(sentence, tokenizer, model))
    X_cls_bert = np.vstack(imdb_df['cls_bert'])
    return X_cls_bert


def get_cls_sentence(sentence, tokenizer, model):
    # Tokenize input sentence and convert to tensor
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True, max_length=512)])

    # Pass input through BERT model and extract embeddings for [CLS] token
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs[0][:, 0, :]

    return cls_embedding.flatten()

# Read in embedding file, 'review' column already preprocessed - bert column: 'cls_bert', glove column: 'encode_glove'
imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')

# Read in orignal file, apply preprocessing to text
imdb_df['review'] = imdb_df['review'].apply(lambda x: text_preprocessing(x))
X = imdb_df['review']
y = imdb_df['sentiment']
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_dev_tfidf, X_test_tfidf = get_tfidf_embedding(X_dev, X_test)
X_dev_bow, X_test_bow = get_bag_of_words_embedding(X_dev, X_test)

# to generate the embedding file locally -> takes a while to run, but only need to run once
get_glove_embedding(imdb_df)
get_bert_embedding(imdb_df)
imdb_df.to_csv("imdb_with_embeddings.csv", index=False)

