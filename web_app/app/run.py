import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")


def get_cat_graphs():
    """
    get all message category counts graphs
    return graph dictionary , the key is the 
    msseage type
    """
    cat2gr={}
    
    for i, col_name in enumerate(message_cat_names):
        val_counts = df[col_name].value_counts()
        vals = val_counts.index.tolist()
        counts = val_counts.values.tolist()
        gr ={
                'data':[
                    Bar(x=vals,
                        y=counts
                        )
                    ],
                'layout':{
                    'title':'Distribution of {}'.format(col_name),
                    'yaxis':{
                        'title':"Count"
                        },
                    'xaxis':{
                        'title':col_name
                        }
                    }
            }
        cat2gr[col_name]=gr
    return cat2gr

#message category name
message_cat_names = df.columns[4:].tolist()
cat2gr = get_cat_graphs()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    
    #print(genre_names)
    #print(genre_counts)
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
        }
    ]

    #add more graphs
    #a dict from column name to graph id
    #the first one is genre

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,message_cat_names=message_cat_names)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    #print(request.args)
    #print(query)

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        message_cat_names=message_cat_names,
        select_cat='related'
    )

#web application to show want plot
@app.route('/display')
def display():
    select_cat = request.args.get('message_cat')
    #print(select_cat)
    graphs=[]
    graphs.append(cat2gr[select_cat])
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('display_graph.html',
                          message_cat_names=message_cat_names,
                          ids=ids,
                          graphJSON=graphJSON,
                          select_cat=select_cat)



def main():
    app.run(host='127.0.0.1', port=5001, debug=True)


if __name__ == '__main__':
    main()
