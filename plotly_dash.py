import pandas as pd
import plotly.graph_objs as go
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html

data=pd.read_csv(r"C:\Users\SUKANYA SAHA\Desktop\Dummy Data\all_stocks_5yr.csv\all_stocks_5yr.csv")
df=pd.DataFrame(data)
df['date']=pd.to_datetime(df['date'])

# external JavaScript files
external_scripts = [
    'https://www.google-analytics.com/analytics.js',
    {'src': 'https://cdn.polyfill.io/v2/polyfill.min.js'},
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.10/lodash.core.js',
        'integrity': 'sha256-Qqd/EfdABZUcAxjOkMi8eGEivtdTkh3b65xCZL4qAQA=',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,external_scripts=external_scripts)
#available_indicators = list(set(df['date'].dt.year))
app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
app.layout = html.Div([
            html.H4('Time series with querter selection'),
            html.Div([
                dbc.Alert([html.H5('Select any option'),
                
                html.P('\n'),
                dcc.RadioItems(
                id='year_group',
                options=[{'label': i, 'value': i} for i in ['Querterly', 'Yearly', 'Monthly', 'All']],
                value='Querterly',
                className='toggle_option',
                labelClassName='switch-radio1',
                labelStyle={'display': 'inline-block'}
                )],
                color='primary',)
                            ], className="mb-3"),
            dbc.Card(
                    dbc.CardBody(
            dcc.Graph(id='indicator-graphic')))   
                        ])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('year_group', 'value'),])
def update_graph(year):
    if year== 'Querterly':
        df=df.groupby(df['date'].dt.to_period('Q')).agg('mean')
    elif year=='Yearly':
        df= df.groupby(df['date'].dt.to_period('Y')).agg('mean')
    elif year== 'Monthly':
        df= df.groupby(df['date'].dt.to_period('M')).agg('mean')
    else:
        df.set_index('date',inplace=True)
    df.index=df.index.astype('datetime64[ns]')

    traces= []
    for col in ['open', 'high', 'low']:
        traces.append(go.Scatter(
                    x=dff.index ,
                    y=dff[col],
                    mode='lines+markers'))
    return {
        'data':traces,
        'layout': go.Layout(title='Title')
        }


if __name__ == '__main__':
    app.run_server()