import json
import plotly
import pandas as pd
import re
import os
from typing import List
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from sqlalchemy import create_engine
import pickle
from pathlib import Path
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.graph_objs import Treemap
from wordcloud import WordCloud

app = Flask(__name__)

def tokenize(text: str) -> List[str]:
    """
    Normalize, tokenize, and lemmatize text string.

    Parameters:
    text : (str)        Text message to be tokenized.

    Returns:
    list                List of clean tokens.
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer that extracts a binary feature indicating whether a text starts with a verb or 'RT'.

    Parameters:
        column (str): The name of the column containing text messages. Default is 'message'.

    Methods:
        starting_verb(text):
            Checks if the first word of the text is a verb or 'RT'.
        
        transform(X):
            Applies the `starting_verb` method to each text entry and returns a sparse matrix of features.
    """
    def __init__(self, column='message'):
        self.column = column

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                continue
            first_word, first_tag = pos_tags[0]
            if first_tag.startswith('VB') or first_word == 'RT':
                return 1
        return 0

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X[self.column]
        else:
            texts = X
        X_tagged = texts.apply(self.starting_verb)
        sparse_matrix = sparse.csr_matrix(X_tagged.values).T
        return sparse_matrix

def load_model(model_filepath: str):
    """
    Loads a saved model from a pickle file.
    
    Args:
        model_filepath (str): The file path to the saved model (.pkl file).
    
    Returns:
        RandomizedSearchCV: The loaded RandomizedSearchCV model pipeline.
    """
    model_path = Path(model_filepath)
    try:
        with model_path.open('rb') as file:
            model = pickle.load(file)
        print(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        raise IOError(f"Failed to load model from {model_path}: {e}")

# load data
engine = create_engine('sqlite:///../data/disaster_database.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = load_model("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    # Fetch data for wordcloud
    binary_columns = df.columns[4:]
    category_counts = df[binary_columns].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)
    counts = category_counts.values.tolist()
    
    # Generate Word Cloud
    text = " ".join(df['message'].astype(str))
    wordcloud_filename = 'wordcloud.png'
    wordcloud_path = os.path.join('static', wordcloud_filename)
    
    # Generate and save the word cloud image
    wordcloud = WordCloud(width=600, height=550, background_color='white', colormap='rainbow').generate(text)
    wordcloud.to_file(wordcloud_path)
    
    # Create Treemap
    treemap = Treemap(
        labels=category_names,
        parents=[""] * len(category_names),
        values=counts,
        marker=dict(
            colors=counts,
            colorscale='Rainbow',
            showscale=True,
            colorbar=dict(
                title='Number of Messages',
                titleside='right',
                tickmode='auto',
                ticks='outside'
            )
        ),
        hoverinfo='label+value',
        textinfo='label+value',
        textfont=dict(
            color='white',
            size=12
        )
    )
    
    treemap_layout = dict(
        title='Distribution of Message Categories',
        margin=dict(t=50, l=25, r=25, b=25),
        height=600,
        width=600
    )
    
    treemap_fig = dict(data=[treemap], layout=treemap_layout)
    treemap_graphJSON = json.dumps(treemap_fig, cls=plotly.utils.PlotlyJSONEncoder)
    wordcloud_image = wordcloud_filename
    
    return render_template('master.html', treemap_graphJSON=treemap_graphJSON, wordcloud_image=wordcloud_image)

@app.route('/go')
def go():
    query = request.args.get('query', '') 

    query_df = pd.DataFrame({'message': [query]})
    classification_labels = model.predict(query_df)[0]

    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()