# COMS 4995 Applied Machine Learning: IMDB Sentiment Analysis Project
## Author: Emma Corbett (ec3745)

## Introduction
IMDb is an online platform where users can provide ratings and reviews for various forms of visual entertainment,
particularly movies. This analysis aims to examine the connection between the review sentiment and the textual
content, with the ultimate goal of predicting binary sentiment ratings from the text. The goal is to develop a
model capable of generating highly accurate sentiment predictions and offering insights into the factors that influence
these emotions in the reviews. This will be achieved by employing and comparing a variety of machine learning and
language models, such as logistic regression, k nearest neighbors, deep neural networks, and bert. By identifying the underlying emotions in movie reviews, this project aims to tackle the
challenge of translating subjective judgment into quantitative metrics.

## Please reach out for the full paper & results.

## Notes on creating embeddings and running locally
The main dataset for this project is IMDB Dataset.csv. It is too large to include in this repository, so can be downloaded here: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

You can run create_embeddings.py with the last code block uncommented out to generate the csv file with bert and glove embeddings.
Once you have the file (also can be downloaded in our google drive), to convert the tensors saved as strings back to tensors, use the following code:

    def return_to_tensor(txt):
        return eval("torch." + txt).numpy()

    imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')
    imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(return_to_tensor)
    X_bert = np.vstack(imdb_df['cls_bert'])
    X_dev_bert, X_test_bert, y_dev_bert, y_test_bert = train_test_split(X_bert, labels, test_size=0.2, random_state=42)

Glove only takes a minute to run locally, so you may be better off re-running it locally than importing it from the file.
To use embeddings by running locally, you can use the following steps:

    from create_embeddings import get_tfidf_embedding, get_bag_of_words_embedding, get_glove_embedding
    
    imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')
    
    X_glove = get_glove_embedding(imdb_df)
    y = imdb_df['sentiment']
    labs = [1 if label == "positive" else 0 for label in y]
    labels = torch.tensor(labs).float().unsqueeze(1)

    
    X_dev, X_test, y_dev, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    X_dev_tfidf, X_test_tfidf = get_tfidf_embedding(X_dev, X_test)
    X_dev_bow, X_test_bow = get_bag_of_words_embedding(X_dev, X_test)

    X_dev_glove, X_test_glove, y_dev_glove, y_test_glove = train_test_split(X_glove, labels, test_size=0.2, random_state=42)

You can look at logistic_regression_model.py for reference.
If you run into versioning issues while trying to run the create_embeddings.py file locally, it's most likely due to torch vs torchtext with bert and glove. To avoid this issue, if you only need to get the glove embeddings, you can comment out the bert code and comment out the import torch line, and that should fix the issue. 
    

    