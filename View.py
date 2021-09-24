# Dash libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc # https://dash.plotly.com/dash-core-components
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import _dendrogram as dendrogram_maker_custom
import plotly.express as px
from Controller import Controller
from wordcloud import WordCloud
import base64
from io import BytesIO

__active_messages__ = {
    # True if message must be shown
    'dataset-front-error':  False,
    'dataset-stk-error':    False,
    'dataset-req-error':    False
}

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container(children=[
    html.Header(id='header',className='p-5 text-center bg-light', children=[
        html.H1('Lara'),
        html.H2('Pareto Front Explorer')
    ]),
    html.Main(id='main_section', children=[
        html.Section(id='messages_section', children=[]),
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(id='data_loading_tab', label='Configuration', children=[
                html.Br(),
                dbc.FormGroup([
                    dbc.Label('Pareto Front File', html_for='pareto_front_input'),
                    dcc.Upload(  # https://dash.plotly.com/dash-core-components/upload
                        id='pareto_front_input',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    dbc.FormText(
                        "Use the example file to format the JSON input file",
                        id='pareto_sub_info',
                        color="secondary"),
                ]),
                # Requirements file upload
                dbc.FormGroup([
                    dbc.Label('Requirements File', html_for='req_input'),
                    dcc.Upload(  # https://dash.plotly.com/dash-core-components/upload
                        id='req_input',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    dbc.FormText(
                        "Use the example file to format the JSON input file",
                        id='req_sub_info',
                        color="secondary"),
                ]),
                # Stakeholders file upload
                dbc.FormGroup([
                    dbc.Label('Stakeholders File', html_for='stk_input'),
                    dcc.Upload(  # https://dash.plotly.com/dash-core-components/upload
                        id='stk_input',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=False
                    ),
                    dbc.FormText(
                        "Use the example file to format the JSON input file",
                        id='stk_sub_info',
                        color="secondary"),
                ]),
                dbc.Button(
                    "Explore",
                    id="explore-button",
                    className="mb-3",
                    color="primary",
                    n_clicks=0
                )
            ]),     # end data_loading_tab
            dcc.Tab(id='data_vis_tab', label='Exploration',  disabled=True, children=[
                html.Section(id='vis_section', children=[
                    dbc.Row(id='dendrogram_row', children=[
                        dcc.Graph(id='dendrogram_plot')
                    ]),
                    dbc.Row(children=[
                        dbc.Col(dcc.Slider(id='threshold_slider', min=0, max=1, value=0.7, step=0.05)),
                        dbc.Col(children=[
                            html.Button('Back', id='back_button', n_clicks=0, disabled=True),
                            dcc.Input(id='desired_cluster', type='number'),
                            html.Button('Dive', id='dive_button', n_clicks=0)
                        ]),
                        dbc.Col(children=[
                            html.Label('Cluster to analyze: '),
                            dcc.Dropdown(
                                id='dropdown_for_wordclouds',
                                options=[],
                                searchable=False,
                                clearable=False
                            )
                        ]),
                    ]),
                    dbc.Row(id='violin_plots', children=[
                        dbc.Col(dcc.Graph(id='violin_plot_profit')),
                        dbc.Col(dcc.Graph(id='violin_plot_cost'))
                    ]),

                    dbc.Row(children=[
                        dbc.Col(children=[
                            dbc.Row(html.Label('Requirements')),
                            dbc.Row(html.Img(id="cloud_plot_req"))
                        ]),
                        dbc.Col(children = [
                            dbc.Row(html.Label('Stakeholders')),
                            dbc.Row(html.Img(id='cloud_plot_stk'))
                        ])
                    ])
                ])
            ]) # end data_vis_tab
        ]) # end tabs
    ]),
])

@app.callback(Output("stk_sub_info", "children"),
    [Input('stk_input', 'contents')])
def stk_input_method(file):
    ctx = dash.callback_context
    if ctx.triggered :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "stk_input" and file :
            controller = Controller.get_instance()
            controller.load_stakeholders_file(file)
            return ['Stakeholders file uploaded']
    else:
        raise PreventUpdate

@app.callback(Output("pareto_sub_info", "children"),
    [Input('pareto_front_input', 'contents')])
def pareto_front_input_method(file):
    ctx = dash.callback_context
    if ctx.triggered :
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "pareto_front_input" and file :
            controller = Controller.get_instance()
            controller.load_pareto_front_file(file)
            return ['Pareto front file uploaded']
    else:
        raise PreventUpdate

@app.callback(
    Output("dendrogram_plot", "figure"),
    Output("violin_plot_profit", "figure"),
    Output("violin_plot_cost", "figure"),
    Output("data_vis_tab", "disabled"),
    Output("back_button", "disabled"),
    Output("dropdown_for_wordclouds", 'options'),
    Output("dropdown_for_wordclouds", 'value'),
    [Input("explore-button", "n_clicks"),
     Input("dive_button", "n_clicks"),
     Input("back_button", "n_clicks"),
     Input("threshold_slider", "value")],
    [State("desired_cluster", "value")])
def update_view(explore_b_clicks, dive_b_clicks, back_b_clicks, threshold=0.7, cluster=0):
    ctx = dash.callback_context
    # TODO: too much spaghetti code, refactor.
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        control = Controller.get_instance()
        back_b_disabled = control.get_depth() <= 1
        print(control.get_depth())
        # TODO: use switch-case structure, but python 3.10 must be used.
        if trigger_id == 'explore-button':
            dendrogram_fig, profit_fig, cost_fig = get_current_visualization(threshold)
            # TODO: call it just once, refactor
            s = list(control.get_clusters(control.get_last_linkage_matrix(), threshold))
            clusters = [{'label': c, 'value': c} for c in  list(map(str, sorted(list(set(s)))))]
            return dendrogram_fig, profit_fig, cost_fig, False, back_b_disabled, clusters, clusters[0]['value'],
        elif trigger_id == "dive_button":
            # TODO: if cluster size too small, disable button
            if control.front_is_loaded():
                if control.get_members_in_cluster(threshold, cluster) > 1:
                    # TODO: if not, an exception and message should be shown
                    control.dive_into_cluster(threshold, cluster)
                dendrogram_fig, profit_fig, cost_fig = get_current_visualization(threshold)
                s = list(control.get_clusters(control.get_last_linkage_matrix(), threshold))
                clusters = [{'label': c, 'value': c} for c in  list(map(str, sorted(list(set(s)))))]
                return dendrogram_fig, profit_fig, cost_fig, dash.no_update, back_b_disabled, clusters, clusters[0]['value']
            else:
                # TODO: implement this part of the method
                pass
        elif trigger_id == "back_button":
            print(control.get_depth())
            if control.front_is_loaded():
                if control.get_depth() > 1:
                    control.go_back()
                dendrogram_fig, profit_fig, cost_fig = get_current_visualization(threshold)
                s = list(control.get_clusters(control.get_last_linkage_matrix(), threshold))
                clusters = [{'label': c, 'value': c} for c in  list(map(str, sorted(list(set(s)))))]
                return dendrogram_fig, profit_fig, cost_fig, dash.no_update, back_b_disabled, clusters, clusters[0]['value']
            else:
                # TODO: implement this part of the method
                pass
        elif trigger_id == "threshold_slider" :
            # TODO: Limit the range in desired_cluster according to the number of clusters
            if control.front_is_loaded():
                dendrogram_fig, profit_fig, cost_fig = get_current_visualization(threshold)
                s = list(control.get_clusters(control.get_last_linkage_matrix(), threshold))
                clusters = [{'label': c, 'value': c} for c in  list(map(str, sorted(list(set(s)))))]
                return dendrogram_fig, profit_fig, cost_fig, dash.no_update, back_b_disabled, clusters, clusters[0]['value']
            else:
                # TODO: implement this part of the method
                pass
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

@app.callback(
    Output("cloud_plot_req", "src"),
    Output("cloud_plot_stk", "src"),
    [Input("dropdown_for_wordclouds", "value")],
    [State("threshold_slider", "value")]
)
def update_wordclouds(cluster, threshold):
    ctx = dash.callback_context
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == "dropdown_for_wordclouds":
            control = Controller.get_instance()
            words = control.get_words_from_cluster(threshold, cluster)
            return  plot_wordcloud(words[1]), plot_wordcloud(words[0])
    raise PreventUpdate

def get_current_visualization(threshold = 0.7):
    dendrogram_fig = get_dendrogram(threshold)
    profit_fig, cost_fig = get_violin_charts(threshold)
    return dendrogram_fig, profit_fig, cost_fig


def get_dendrogram(threshold: float):
    control = Controller.get_instance()
    linkage_matrix = control.get_last_linkage_matrix()
    labs = control.get_data_indexes()
    ct = threshold * max(linkage_matrix[:, 2])
    dn = dendrogram_maker_custom.create_dendrogram(linkage_matrix, color_threshold=ct, labels=list(labs))
    return dn


def get_violin_charts(threshold: float):
    control = Controller.get_instance()
    # TODO: redo this, get_clusters always call the fcluster function, which is inefficient
    clusters = control.get_clusters(control.get_last_linkage_matrix(), threshold)
    dataset = control.get_current_front()
    dataset['cluster'] = list(map(str, clusters))
    profit_fig = px.violin(dataset, x='cluster', y='profit', title='Profit', points='all')
    cost_fig = px.violin(dataset, x='cluster', y='cost', title='Cost', points='all')
    return profit_fig, cost_fig


def plot_wordcloud(data):
    d = dict(zip(data[0], data[1]))
    wc = WordCloud(background_color='white', width=480, height=360)
    wc.fit_words(d)
    img = BytesIO()
    wc_img = wc.to_image()
    wc_img.save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


if __name__ == '__main__':
    app.run_server(debug=True)