import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose


st.set_page_config(page_title="Electric Consumption", page_icon=":electric_plug:", layout="wide")


filepath = r'final_data.csv'

energycon_df = pd.read_csv(filepath)
energycon_df['Datetime'] = pd.to_datetime(energycon_df['Datetime'])  # Convert the date column to datetime
energycon_df.set_index('Datetime', inplace=True)
energycon_df.sort_values(by=['Datetime'], inplace=True, ascending=True)

# Checking if there are differences in the zero values of PFT and PFT
zero_mwt = energycon_df.loc[energycon_df.loc[:,'MWT'] == 0].index
zero_pft = energycon_df.loc[energycon_df.loc[:,'PFT'] == 0].index



def create_dataset(dataset, look_back=1):
    dataX = []  # Initializing as a list
    dataY = []  # Initializing as a list

    for i in range(len(dataset) - look_back):
        dataX.append(dataset.iloc[i:(i + look_back)].values)
        dataY.append(dataset.iloc[i + look_back])

    # Convert the lists to DataFrames
    dataX = pd.DataFrame(np.array(dataX).reshape(-1, look_back))
    dataY = pd.Series(dataY)

    return dataX, dataY

# Imputing zero values
for index in zero_mwt:
    energycon_df.loc[index,'MWT'] = np.random.normal(
        energycon_df.loc[:, 'MWT'].mean(),
        energycon_df.loc[:, 'MWT'].std()
        )
energycon_df.loc[index,'PFT'] = energycon_df.loc[:, 'PFT'].mean()

MWT_data = energycon_df
MWT_data.index.freq = 'H'

st.title("Energy Consumption Projection")

# Load your data and perform necessary preprocessing here

def mwt_time_series_plot():
    st.subheader('MWT Time Series Plot', 
               divider='gray',
               help="This shows the MegaWattTotal of the dataset")
    fig_mwt = go.Figure()
    fig_mwt.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MWT'], mode='lines', name='MWT', line=dict(color='dodgerblue')))
    fig_mwt.update_layout(
        xaxis_title='Month',
        yaxis_title='MWT',
        
        
    )
    st.plotly_chart(fig_mwt, use_container_width=True)

def mwt_distribution():
    st.subheader('MWT Distribution', 
               divider='gray',
               help="This shows the MegaWatt Total normal distribution histogram of the dataset. The result indicates that the distribution was good")
    plotly_fig_dist = go.Figure(data=[go.Histogram(x=MWT_data['MWT'], nbinsx=20, marker=dict(color='dodgerblue'))])
    plotly_fig_dist.update_layout(
        xaxis_title='MWT Values',
        yaxis_title='Frequency'
    )
    st.plotly_chart(plotly_fig_dist, use_container_width=True)

def roll_stats():
    st.subheader('Rolling Statistics', 
               divider='gray',
               help="This shows the rolling statistics like mean and Standard deviation. The plot displays the time series plot of the actual data and the rolling statistics of the dataset")
    rolling_mean = MWT_data['MWT'].rolling(10).mean()
    rolling_std = MWT_data['MWT'].rolling(10).std()

    fig_rolling = go.Figure()

    fig_rolling.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MWT'], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
    fig_rolling.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, mode='lines', name='Rolling Mean', line=dict(color='limegreen')))
    fig_rolling.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std, mode='lines', name='Rolling STD', line=dict(color='indianred')))

    fig_rolling.update_layout(
        
        xaxis_title='Date',
        yaxis_title='MWT',
        xaxis=dict(tickformat="%b")  # Show all months on x-axis
    )
    st.plotly_chart(fig_rolling, use_container_width=True)

def pft_time_series_plot():
    st.subheader('PFT Time Series Plot', 
               divider='gray',
               help="This shows the Power Factor Total of the dataset using Time Series plot. ")
    fig_pft = go.Figure()
    fig_pft.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['PFT'], mode='lines', name='PFT', line=dict(color='green')))
    fig_pft.update_layout(
        xaxis_title='Month',
        yaxis_title='PFT',
        
    )
    st.plotly_chart(fig_pft, use_container_width=True)

def trend_mwt():
    st.subheader('Trend of MWT', 
               divider='gray',
               help="This shows the Trend of MegaWatt Total in the dataset using time series plot")
    decomposition = seasonal_decompose(MWT_data['MWT'], model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=MWT_data.index, y=trend, mode='lines', name='Trend', line=dict(color='dodgerblue')))
    fig_trend.update_layout(title='Trend of MWT', xaxis_title='Month', yaxis_title='Trend')
    st.plotly_chart(fig_trend, use_container_width=True)


original_df_perhour = MWT_data
resampled_df_perday = MWT_data.resample('D').sum()
resampled_df_weekly = MWT_data.resample('7D').sum()
resampled_df_monthly = MWT_data.resample('M').sum()

def perHour():
    st.subheader('MWT Hourly', 
               divider='gray',
               help="This shows the MegaWatt Total of the dataset in hours (January - December 2022) using Time series plot")
    fig_original = go.Figure()
    fig_original.add_trace(go.Scatter(x=original_df_perhour.index, y=original_df_perhour['MWT'], mode='lines', name='Original Time Series Data', line=dict(color='dodgerblue')))
    fig_original.update_layout( xaxis_title='Date', yaxis_title='MWT',height=500)
    st.plotly_chart(fig_original, use_container_width=True)

def perDay():
    st.subheader('MWT Daily', 
               divider='gray',
               help="This shows the MegaWatt Total of the dataset in days (January - December 2022) using Time series plot")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=resampled_df_perday.index, y=resampled_df_perday['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_daily.update_layout(title='MWT Daily', xaxis_title='Date', yaxis_title='MWT')
    st.plotly_chart(fig_daily, use_container_width=True)

def perWeek():
    st.subheader('MWT Weekly', 
               divider='gray',
               help="This shows the MegaWatt Total of the dataset in weeks (January - December 2022) using Time series plot")
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(x=resampled_df_weekly.index, y=resampled_df_weekly['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_weekly.update_layout(title='MWT Weekly', xaxis_title='Date', yaxis_title='MWT')
    st.plotly_chart(fig_weekly, use_container_width=True)

def perMonth():
    st.subheader('MWT Monthly', 
               divider='gray',
               help="This shows the MegaWatt Total of the dataset in months (January - December 2022) using Time series plot")
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(x=resampled_df_monthly.index, y=resampled_df_monthly['MWT'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
    fig_monthly.update_layout(title='MWT Monthly', xaxis_title='Date', yaxis_title='MWT')
    st.plotly_chart(fig_monthly, use_container_width=True)

    
top_left_column, top_right_column = st.columns((1, 1))

center_left_column, center_right_column = st.columns((1, 1))

bottom_left_column_1, bottom_right_column_2 = st.columns(2)

bottom_left_column_3, bottom_right_column_4 = st.columns(2)

with top_left_column:
    mwt_time_series_plot()
    

with top_right_column:
    roll_stats()

with center_left_column:
    column_center_left_1, column_center_left_2 = st.columns(2)

    with column_center_left_1:
        mwt_distribution()
    with column_center_left_2:
        pft_time_series_plot()

with center_right_column:
    trend_mwt()

with bottom_left_column_1:
    perMonth()
with bottom_right_column_2:
    perWeek()
with bottom_left_column_3:
    perDay()
with bottom_right_column_4:
    perHour()