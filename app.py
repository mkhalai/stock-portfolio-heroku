
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def optimize(data):
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    # Optimise for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    weights = np.array(list(cleaned_weights.values()))
    folio = (data.iloc[:,:]*weights).sum(axis=1)
    perf = ef.portfolio_performance(verbose=True)
    return weights, folio,perf

def plotstocks(df):
    """Plot the stocks in the dataframe df"""
    figure = go.Figure()
    alpha=0.3
    lw=1
    for stock in df.columns.values:
        if stock == 'portfolio':
            alpha=1
            lw = 3
        else:
            alpha=0.3
            lw=1
        figure.add_trace(go.Scatter(
            x=df.index.values,
            y=df[stock],
            name=stock,
            mode='lines',
            opacity=alpha,
            line={'color':'Navy','width': lw}
        ))
    figure.update_layout(height=600,width=800,
                         xaxis_title='Date',
                         yaxis_title='Relative growth %',
                         title='Relative Growth of optimized portfolio')
    figure.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return figure


snp = pd.read_csv('snp10.csv',index_col=0)

st.write("""
 # S&P Top 10 Stock portfolios
 ## **Create** your own ***optimized*** stock portfolio
""")

symbols = snp.columns.values
check_boxes = [st.sidebar.checkbox(symbol, key=symbol) for symbol in symbols]
checked = [symbol for symbol, checked in zip(symbols, check_boxes) if checked]

if len(checked) == 0:
    st.write(" ## ERROR: Select some stocks")
data = snp[checked]
dates = data.index.values
weights,folio,performance = optimize(data)
performance = np.round(performance,2)
data['portfolio'] = folio
growth = (data/data.iloc[0,:]-1)*100

st.plotly_chart(plotstocks(growth))
st.write(
    """
    ## Portfolio Performance
    ### Anualized Return: ***{}%***
    ### Anualized Volatility: ***{}%***
    ### Sharpe Ratio: ***{}***
    """.format(np.round(performance[0]*100,2),
               np.round(performance[1]*100,2),
               np.round(performance[2],2))
)


w = np.array(weights)
s = np.array(checked)

pie = go.Figure(data = [go.Pie(labels = s[w>0], values=w[w>0])])
pie.update_layout(
    title = "Resource allocation"
)
st.plotly_chart(pie)

period = st.slider('rolling period',min_value=1,max_value=100,value=5,step=1)
folio_daily_returns = data.pct_change().iloc[:,-1]
rolling_volatility = data.pct_change().rolling(period).std().iloc[:,-1]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=dates,
                          y=folio_daily_returns,
                          opacity=0.6,
                          line={'color':'Navy','width':1}))
fig3.add_trace(go.Scatter(x=dates,y=rolling_volatility))
st.write(
    """## Portfolio daily returns and {} period rolling volatility""".format(period)
)
fig3.update_layout(xaxis_title = 'Date',
                   yaxis_title = 'Returns',
                   height = 600,
                   width=800
                   )
st.plotly_chart(fig3)
