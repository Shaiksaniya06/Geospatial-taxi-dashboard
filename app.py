import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# -------------------------
# LOAD DATA
# -------------------------
DATA_PATH = "data/yellow_tripdata_2025-01.parquet"

df = pd.read_parquet(DATA_PATH)

# Sample to avoid memory issues
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

# Convert pickup time
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['date'] = df['tpep_pickup_datetime'].dt.date

# Create fake coordinates (for visualization only)
np.random.seed(42)
df['lat'] = 40.55 + np.random.rand(len(df)) * 0.35
df['lon'] = -74.15 + np.random.rand(len(df)) * 0.35

# -------------------------
# CLUSTERING
# -------------------------
coords = df[['lat', 'lon']]
scaled = StandardScaler().fit_transform(coords)

db = DBSCAN(eps=0.15, min_samples=40).fit(scaled)
df['cluster'] = db.labels_

# -------------------------
# DASH APP
# -------------------------
app = Dash(__name__)
app.title = "NYC Yellow Taxi Dashboard"


# -------------------------
# LAYOUT
# -------------------------
app.layout = html.Div([ 

    html.H2("NYC Yellow Taxi Geospatial Dashboard",
            style={'textAlign': 'center', 'color': '#1f77b4'}),
       
   
    

   html.Div([ 

 

        # FILTER PANEL
        html.Div([

            html.H4("Filters"),

            html.Label("Pickup Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df['tpep_pickup_datetime'].min(),
                max_date_allowed=df['tpep_pickup_datetime'].max(),
                start_date=df['tpep_pickup_datetime'].min(),
                end_date=df['tpep_pickup_datetime'].max(),
                display_format="DD/MM/YYYY"
            ),

            html.Br(), html.Br(),

            html.Label("Min Passengers"),
            dcc.Slider(1, 6, 1, value=1,
                       marks={i: str(i) for i in range(1,7)},
                       id='passenger-slider'),

            html.Br(), html.Br(),

            html.Label("Trip Distance (miles)"),
            dcc.RangeSlider(0, 20, 1, value=[0, 20],
                            id='distance-slider'),

            html.Br(), html.Br(),

            html.Label("Time of Day"),
            dcc.Checklist(
                id='time-of-day',
                options=[
                    {'label': ' Morning', 'value': 'morning'},
                    {'label': ' Afternoon', 'value': 'afternoon'},
                    {'label': ' Evening', 'value': 'evening'},
                    {'label': ' Night', 'value': 'night'}
                ],
                value=['morning','afternoon','evening','night']
            )

        ], style={
            'width': '22%',
            'padding': '12px',
            'backgroundColor': '#f4f6f9',
            'borderRadius': '10px',
            'boxShadow': '2px 2px 6px lightgrey'
        }),


        # CHART AREA
        html.Div([

            html.Div([
                dcc.Graph(id='map-graph', config={'displayModeBar': False},
                          style={'width': '50%', 'height': '300px', 'display': 'inline-block'}),

                dcc.Graph(id='time-chart', config={'displayModeBar': False},
                          style={'width': '50%', 'height': '300px', 'display': 'inline-block'})
            ]),

            html.Div([
                dcc.Graph(id='distance-chart', config={'displayModeBar': False},
                          style={'width': '50%', 'height': '300px', 'display': 'inline-block'}),

                dcc.Graph(id='passenger-chart', config={'displayModeBar': False},
                          style={'width': '50%', 'height': '300px', 'display': 'inline-block'})
            ])

        ], style={
            'width': '76%',
            'paddingLeft': '10px'
        })

    ], style={'display': 'flex'})

], style={'fontFamily': 'Segoe UI'})


# -------------------------
# CALLBACK
# -------------------------
@app.callback(
    Output('map-graph','figure'),
    Output('time-chart','figure'),
    Output('distance-chart','figure'),
    Output('passenger-chart','figure'),
    Input('date-range','start_date'),
    Input('date-range','end_date'),
    Input('passenger-slider','value'),
    Input('distance-slider','value'),
    Input('time-of-day','value')
)
def update_dashboard(start, end, passengers, distance, tod):

    dff = df.copy()

    dff = dff[(dff['tpep_pickup_datetime'] >= start) &
              (dff['tpep_pickup_datetime'] <= end)]

    dff = dff[dff['passenger_count'] >= passengers]
    dff = dff[(dff['trip_distance'] >= distance[0]) &
              (dff['trip_distance'] <= distance[1])]

    def period(h):
        if 6 <= h < 12: return 'morning'
        if 12 <= h < 18: return 'afternoon'
        if 18 <= h < 24: return 'evening'
        return 'night'

    dff['period'] = dff['hour'].apply(period)
    dff = dff[dff['period'].isin(tod)]

    # MAP
    map_fig = px.scatter_mapbox(
        dff, lat='lat', lon='lon', color='cluster',
        zoom=9, height=300,
        title="Pickup Hotspots"
    )
    map_fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=30,b=0))

    # TIME
    time_df = dff.groupby('hour').size().reset_index(name='Trips')
    time_fig = px.line(time_df, x='hour', y='Trips', markers=True,
                       title="Trips by Hour")
    time_fig.update_layout(margin=dict(l=30,r=10,t=30,b=30))

    # DISTANCE
    dist_fig = px.histogram(dff, x='trip_distance', nbins=30,
                             title="Trip Distance Distribution")
    dist_fig.update_layout(margin=dict(l=30,r=10,t=30,b=30))

    # PASSENGERS
    pass_df = dff['passenger_count'].value_counts().reset_index()
    pass_df.columns = ['Passengers','Trips']
    pass_fig = px.bar(pass_df, x='Passengers', y='Trips',
                      title="Trips by Passenger Count")
    pass_fig.update_layout(margin=dict(l=30,r=10,t=30,b=30))

    return map_fig, time_fig, dist_fig, pass_fig


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)


















