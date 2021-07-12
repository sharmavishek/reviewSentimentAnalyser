# Importing the libraries
import pickle
import pandas as pd
import numpy as np
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
project_name = "Review's Sentiments Analyser"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    
    
    scrappedReviews = pd.read_csv('scrapped.csv')
    
    file = open("my_model.pkl", 'rb') 
    pickle_model = pickle.load(file)

    file = open("myfeature.pkl", 'rb') 
    vocab = pickle.load(file)
        
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)


def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        
        dbc.Jumbotron(
            
            
            
                [  
                    
                    
                    dbc.Container([
                        html.Div(
                        html.Img(src=app.get_asset_url('pie.jpg'),
                                 style={
                                     'margin-bottom':'3%'
                                     }
                                 )
                        
                        ),
                    html.Div(
                        html.Img(src=app.get_asset_url('j.png')))
                    ]
                    
                    ),
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review",value='Good product', style = {'height': '150px'}),
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Search a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in scrappedReviews.reviews],
                    value = scrappedReviews.reviews[0],
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px',}
                        ),
                    dbc.Button("Submit", color="danger", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    html.Div(id = 'result1')
                    ],
                style = { 'background':'orange','box-shadow':'-15px  15px 15px brown',},
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    global app
    global project_name
    load_model()
    
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server(host = '0.0.0.0')
    app = None
    project_name = None
if __name__ == '__main__':
    main()