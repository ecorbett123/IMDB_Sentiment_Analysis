# COMS 4995 Applied Machine Learning: IMDB Sentiment Analysis Project
## Authors: Alan Wong (ajw2252), Anna Micros (am6529), Emma Corbett (ec3745), Nuneke Kwetey (nfk2108), Xiqian Yuan (xy2655)
The main dataset for this project is IMDB Dataset.csv. It is too large to include in this repository, so can be downloaded here: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

You can run create_embeddings.py with the last code block uncommented out to generate the csv file with bert and glove embeddings.
Once you have the file (also can be downloaded in our google drive), to convert the tensors saved as strings back to tensors, use the following code:

    def return_to_tensor(txt):
        return eval("torch." + txt).numpy()

    imdb_df = pd.read_csv('imdb_with_glove_bert_embeddings.csv')
    imdb_df['cls_bert'] = imdb_df['cls_bert'].apply(return_to_tensor)