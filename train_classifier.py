import sys
import os
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from typing import Tuple, List, Union
from scipy import sparse
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Download nltk addons if needed
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'punkt_tab'])
nltk.download('averaged_perceptron_tagger_eng')


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Load SQLite database into pd.DataFrame. Split features and targets.

    Parameters:
        database_filepath (str) : path to SQlite database

    Returns:
        X (pd.DataFrame)   - features dataframe
        Y (pd.DataFrame)   - targets dataframe
        target_names (list) - names of target variables

    """
    if not os.path.exists(database_filepath):
        raise FileNotFoundError(f"Database file not found at: {database_filepath}")
    
    try:
        engine = create_engine(f'sqlite:///{database_filepath}')
        df = pd.read_sql_table("disaster_response", engine)
        
        print(f"Successfully loaded table 'disaster_response' from '{database_filepath}'.")

    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
    
    df_noid = df.drop(columns=["id", "original", "genre"])
    
    feature_cols = ["message"]
    X = df_noid[feature_cols]
    Y = df_noid.drop(columns=feature_cols)

    target_names = list(Y.columns)

    return X, Y, target_names


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


def build_model() -> RandomizedSearchCV:
    """
    Builds and returns a RandomizedSearchCV model pipeline with text processing
    and starting verb feature extraction using XGBoost as the classifier.

    Returns:
        RandomizedSearchCV: Configured randomized search model.
    """
    # 1 Text processing pipeline
    text_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
    ])

    # 2 Starting verb extractor pipeline
    starting_verb_pipeline = Pipeline([
        ('starting_verb', StartingVerbExtractor(column='message'))
    ])

    # 3 Join pipelines using ColumnTransformer
    feature_union = ColumnTransformer([
        ('text_pipeline', text_pipeline, 'message'),
        ('starting_verb', starting_verb_pipeline, 'message')
    ])

    # 4 Complete pipeline with MultiOutputClassifier using XGBoost
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', MultiOutputClassifier(
            XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        ))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1, 1)],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [3, 4],
        'clf__estimator__learning_rate': [0.1, 0.01],
        'clf__estimator__min_child_weight': [1],
        'clf__estimator__subsample': [0.8, 1]
    }

    cv = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=16,
        cv=3,
        verbose=3,
        n_jobs=-1,
        scoring='f1_micro',
        random_state=42
    )

    return cv


def evaluate_model(
    model: RandomizedSearchCV,
    X_test: Union[pd.DataFrame, np.ndarray],
    Y_test: Union[pd.DataFrame, np.ndarray],
    category_names: List[str]
) -> None:
    """
    Evaluate a multi-output XGBoost classifier model by printing classification reports and accuracies for each category.

    Args:
        model (RandomizedSearchCV): Trained multi-output classification model.
        X_test (pd.DataFrame or np.ndarray): Test features.
        Y_test (pd.DataFrame or np.ndarray): True labels for the test set.
        category_names (List[str]): Names of target categories.

    Raises:
        ValueError: If Y_test or y_pred are not DataFrame, Series, or ndarray, or if their shapes mismatch.
    """
    if isinstance(Y_test, np.ndarray):
        Y_test = pd.DataFrame(Y_test, columns=category_names)
    elif isinstance(Y_test, pd.Series):
        Y_test = Y_test.to_frame()
    elif not isinstance(Y_test, pd.DataFrame):
        raise ValueError("Y_test should be a pandas DataFrame, Series, or NumPy array.")

    y_pred = model.predict(X_test)

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=category_names)
    elif isinstance(y_pred, pd.Series):
        y_pred = y_pred.to_frame()
    elif not isinstance(y_pred, pd.DataFrame):
        raise ValueError("y_pred should be a pandas DataFrame, Series, or NumPy array.")

    Y_test = Y_test.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    if Y_test.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: Y_test shape {Y_test.shape} vs y_pred shape {y_pred.shape}")

    accuracies = {}

    for category in category_names:
        print(f"--- Category: {category} ---")
        report = classification_report(Y_test[category], y_pred[category], zero_division=0)
        print(report)
        accuracy = (Y_test[category] == y_pred[category]).mean()
        accuracies[category] = accuracy
        print(f"Accuracy for {category}: {accuracy:.4f}\n")

    accuracies_df = pd.DataFrame.from_dict(accuracies, orient='index', columns=['Accuracy'])
    accuracies_df = accuracies_df.sort_values(by='Accuracy', ascending=False)

    print("=== Overall Accuracy Across All Categories ===")
    print(accuracies_df)

    try:
        print("\nBest Parameters:", model.best_params_)
    except AttributeError:
        print("\nBest Parameters: Not available (Model is not a GridSearchCV instance)")


def save_model(model: XGBClassifier, model_filepath: str) -> None:
    """
    Save the XGBoost classifier model to a specified file path in pickle format.

    Args:
        model (XGBClassifier): Trained XGBoost classifier model.
        model_filepath (Union[str, Path]): Destination file path for the saved model.
    """
    model_path = Path(model_filepath)
    try:
        with model_path.open('wb') as file:
            pickle.dump(model, file)
        print(f"Model successfully saved to {model_path}")
    except Exception as e:
        raise IOError(f"Failed to save model to {model_path}: {e}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
