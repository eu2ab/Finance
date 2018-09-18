# https://medium.com/@plotlygraphs/introducing-dash-5ecf7191b503
# https://plot.ly/~jackp/17561/import-dash-from-dashdependencies-impor/#/
# https://plot.ly/products/dash/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import datetime as dt
import json
from GetData import QUANDL_price

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

#   Build data frame containing positions/information
df = pd.DataFrame({'Ticker': ['AAPL', 'GOOGL', 'TSLA'],
                   'Cost Basis': [100, 100, 100],
                   'Date of Purchase': [dt.date(2016, 1, 1), dt.date(2016, 1, 1), dt.date(2016, 1, 1)],
                   'Delta': [15, 15, 15]})  # Build initial dataframe
df['Stock Price'] = QUANDL_price('AAPL')
