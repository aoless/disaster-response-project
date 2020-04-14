import json

import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from joblib import load
from plotly.graph_objs import Bar
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine

from disaster_response.data.process_data import load_data
from disaster_response.data.process_data import clean_data
from disaster_response.models.train_classifier import tokenize


app = Flask(__name__)

# load data
df = load_data("../data/disaster_messages.csv", "../data/disaster_categories.csv")
df_cleaned = clean_data(df)

# load model
model = load("../models/model.pkl")
model.verbose = False

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    num_of_feature_cols = 4
    categories_df = df_cleaned[df_cleaned.columns[num_of_feature_cols:]]
    
    categories_dist = categories_df.sum(axis=0)
    categories_names = [category.replace("_", " ") for category in categories_dist.index]

    colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
        'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
        'rgba(190, 192, 213, 1)']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=categories_names[3:],
                    y=list(categories_dist.values)[3:],
                )
            ],

            'layout': {
                'title': 'Categories dist',
                'margin': {'b': 160},
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'tickangle': 45,
                    'title': "Count"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_counts,
                    y=genre_names,
                    orientation='h',
                    marker=dict(
                        color='rgba(246, 78, 139, 0.6)',
                        line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
                    ),
                )
            ],

            'layout': {
                'title': 'Some other graph',
                'yaxis': {
                    'title': "Genre"
                },
                'xaxis': {
                    'title': "Count"
                },
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(classification_labels)
    classification_results = dict(zip(df_cleaned.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()