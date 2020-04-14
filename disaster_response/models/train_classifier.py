import nltk
import numpy as np
import pickle
import pandas as pd
import sys

from joblib import load, dump
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine


def load_database(database_filepath):
    """Reads database from database_filepath and converts it to pandas DataFrame"""
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("DisasterResponse", engine)

    return df


def extract_features_and_labels(df):
    """Extracts features and labels matrix from df"""
    num_of_feature_cols = 4

    X = df.message.values
    Y = df[df.columns[num_of_feature_cols:]]
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """Splits text into tokens and applies lemmatisation on every one of it"""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    clean_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]

    return clean_tokens


def build_model():
    """Creates sklearn nlp pipeline and search for the best parameters"""
    pipeline = Pipeline([
        ("c_vect", CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2), max_df=0.95)),
        ("tfidf", TfidfTransformer(use_idf=True, smooth_idf=True)),
        ("clf", MultiOutputClassifier(RandomForestClassifier(verbose=1, n_jobs=6))),
    ])

    parameters = {
        "clf__estimator__n_estimators": [100, 200],
        "clf__estimator__min_samples_split": [4, 8]
    }

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates how accurate is a model with it's predictions"""
    Y_pred = model.predict(X_test)
    show_scores(Y_pred, Y_test, category_names)


def extract_macro_avg(report):
    """Extracts macro avg fscore from sklearn's classification report"""
    for item in report.split("\n"):
        if "macro avg" in item:
            return float(item.strip().split()[4])


def show_scores(predicted_values, real_values, classes_names):
    """Shows classifiaction report for every category and counts mean fscore macro avg"""
    macro_avg_list = []
    
    for i in range(1, len(classes_names)):
        report = classification_report(
            real_values.iloc[:, i].values,
            predicted_values[:, i],
            zero_division=1)
        macro_avg_list.append(extract_macro_avg(report))
        print("Category:", classes_names[i], "\n", report)
        
    overall_avg_score = sum(macro_avg_list) / len(macro_avg_list)
    print(f"Overral average score: {overall_avg_score:.3}")


def save_model(model, model_filepath):
    """Saves model to given filepath"""
    dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df = load_database(database_filepath)
        X, Y, category_names = extract_features_and_labels(df)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model.\nThis may take a while...')
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