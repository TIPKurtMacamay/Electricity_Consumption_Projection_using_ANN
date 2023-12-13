import streamlit as st
from datetime import datetime
from Home import MWT_data
import pages.Long_Term_Projection as ltf
import Home as e_app
import plotly.figure_factory as ff
import plotly.graph_objects as go
from tensorflow import keras
import numpy as np

import random

def short_term_projection():
    st.title("Short Term Projection")
    # Your long-term projection page content goes here



if __name__ == "__main__":
    short_term_projection()
    

selected_on_calendar = 0

st.sidebar.title("Settings")
st.sidebar.divider()

def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    with st.container():
        fig = go.Figure(
            go.Indicator(
                value=indicator_number,
                mode="gauge+number",
                domain={"x": [0, 1], "y": [0, 1]},
                number={
                    "suffix": indicator_suffix,
                    "font.size": 30,
                    
                },
                
                gauge={
                    "axis": {"range": [0, max_bound], "tickwidth": 5},
                    "bar": {"color": indicator_color},
                },
                title={
                    "text": indicator_title,
                    "font": {"size": 28},
                },
                
            )
        )
        fig.update_layout(
            # paper_bgcolor="lightgrey",
            height=180,
            margin=dict(l=1, r=1, t=1, b=1, pad=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        



plot_top_left = st.columns([10])[0]
bottom_info_left, bottom_info_right = st.columns([9,1])
 

        

# Display the date input
st.markdown('<div class="center">', unsafe_allow_html=True)
default_date = datetime(2022, 1, 1)
min_date = datetime(2022, 1, 1)
max_date = datetime(2022, 12, 31)

st.sidebar.subheader('Choose a date', 
               divider='gray',
               help="This changes the line chart above based on the date you selected. The 6 metrics below will also change based on your selected date")

selected_date = st.sidebar.date_input("Select a date",
                            label_visibility="collapsed",
                            value=default_date,
                            min_value=min_date,
                            max_value=max_date)
st.markdown('</div>', unsafe_allow_html=True)
if selected_date:
    month_name = selected_date.strftime('%B')
    
# Reference date in 2022
reference_date_2022 = datetime(2022, 1, 1, 0, 0, 0)  # Specify a datetime for the reference date

# Calculate the difference between the selected date and the reference date in 2022
date_difference = selected_date - reference_date_2022.date() 

# Extract the number of days from the timedelta
number_of_days = date_difference.days

selected_on_calendar = (number_of_days + 1) * 24
        

st.sidebar.subheader('Choose a model', 
               divider='gray',
               help="This changes the line chart projection result above based on the deep learning model you selected. It also changes the 6 metrics below based on your selection")

option = st.sidebar.selectbox(' ',("MLP", "LSTM", "GRU"), 
                              key="short_term_models")


month_selected = ltf.updated_dict_month_values[month_name]
month_max_month = list(ltf.updated_dict_month_values.values())
month_day_index = list(ltf.updated_dict_month_values.values()).index(month_selected)

min_plot_value = month_max_month[month_day_index - 1]

from sklearn.preprocessing import MinMaxScaler

# Feature scaling
scaler = MinMaxScaler()
MWT_data['MinMax'] = scaler.fit_transform(MWT_data['MWT'].values.reshape(-1, 1))

# Plotly figure for 'MinMax' Time Series
fig_minmax = go.Figure()
fig_minmax.add_trace(go.Scatter(x=MWT_data.index, y=MWT_data['MinMax'], mode='lines', name='Time Series Data', line=dict(color='dodgerblue')))
fig_minmax.update_layout(
    title='Normalized MWT Time Series Plot',
    xaxis_title='Date',
    yaxis_title='MWT'
)

# Split the data into training and testing DataFrames
data = MWT_data['MinMax']

train_size = int(len(data) * 0.8)  # 80% of the data for training

train = data.iloc[:train_size]  # Select the first 80% for training
test = data.iloc[train_size:]   # Select the remaining 20% for testing

# Plotly figure for Train-Test Split
fig_train_test = go.Figure()
fig_train_test.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Training Data', line=dict(color='dodgerblue')))
fig_train_test.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Testing Data', line=dict(color='indianred')))
fig_train_test.update_layout(
    title='80/20 Train-Test Split',
    xaxis_title='Time',
    yaxis_title='Value'
)


look_back = 24  # Number of previous hours to use for prediction
trainX, trainY = e_app.create_dataset(train, look_back)
testX, testY = e_app.create_dataset(test, look_back)


if option == "MLP":
    # Load your Keras model
    model = keras.models.load_model('best_model_mlp.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()

    
elif option == "LSTM":
    # Load your Keras model
    model = keras.models.load_model('best_model_lstm_latest.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()

    
elif option == "GRU":
    # Load your Keras model
    model = keras.models.load_model('best_model_gru.h5')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    trainPredict = trainPredict.flatten()
    testPredict = testPredict.flatten()

       
df_total_predictions = np.array([])
df_total_predictions = np.append(trainPredict, df_total_predictions)

for _ in range(48):
    df_total_predictions = np.append(df_total_predictions, 0)
    
df_total_predictions = np.append(df_total_predictions, testPredict)

with plot_top_left:
    st.subheader('One Day Projection Line Chart', 
               help="This line chart display the Actual and Projection value based on your selected deep learning model. It also display the Time (range of hours 00:00 - 24:00) and value of MWT.",
               divider='gray')
    if option == "MLP":
        
        min_calendar_val = selected_on_calendar - 24
        # Plotting actual vs predicted data in Plotly

        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=data[min_calendar_val:selected_on_calendar], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=df_total_predictions[min_calendar_val:selected_on_calendar], mode='lines', name='Predicted Data', line=dict(color='indianred')))
        
        fig_january.update_layout(
            
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)
    elif option == "LSTM":
        min_calendar_val = selected_on_calendar - 24
         # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=data[min_calendar_val:selected_on_calendar], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=df_total_predictions[min_calendar_val:selected_on_calendar], mode='lines', name='Predicted Data', line=dict(color='indianred')))

        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Day Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

    elif option == "GRU":
        min_calendar_val = selected_on_calendar - 24
        # Plotting actual vs predicted data in Plotly
        fig_january = go.Figure()

        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=data[min_calendar_val:selected_on_calendar], mode='lines', name='Actual Data', line=dict(color='dodgerblue')))
        fig_january.add_trace(go.Scatter(x=data.index[min_calendar_val:selected_on_calendar], y=df_total_predictions[min_calendar_val:selected_on_calendar], mode='lines', name='Predicted Data', line=dict(color='indianred')))
        
        fig_january.update_layout(
            title='Actual vs Predicted Data for January 2022 (One Day Prediction)',
            xaxis_title='Time',
            yaxis_title='Value',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Display Plotly figure using st.plotly_chart()
        st.plotly_chart(fig_january, use_container_width=True)

st.sidebar.subheader('Choose an hour', 
               divider='gray',
               help="This changes the line chart projection result above based on your selected hour. It also changes the 6 metrics below based on your selection. The slider has a range from 1 to 24, it indicates the number of hours a day")

selected_value_hour = st.sidebar.select_slider(
            "Select an hour:",
            options=list(range(1, 25)),
            value=1,
            key="slidebar_hour")
    

        # Display the selected value
st.sidebar.write(f"Selected hour: {selected_value_hour} :00")

with bottom_info_right:
    st.divider()
    number = df_total_predictions[min_calendar_val + (selected_value_hour - 1)]
    truncated_number = float(f'{number:.3f}')

    total_saved_mwt =  data[min_calendar_val + (selected_value_hour - 1)] - truncated_number
    truncated_saved_mwt = float(f'{total_saved_mwt:.3f}')

    percentage_saved_mwt = (total_saved_mwt / data[min_calendar_val + (selected_value_hour - 1)]) * 100
    truncated_percentage_saved_mwt = float(f'{percentage_saved_mwt:.3f}')

    bottom_info_right.metric("Predicted MWT", truncated_number,
                            help="This indicator displays the Predicted MWT value with decimal places. The value of MWT was from the gauge indicator.")
    bottom_info_right.metric("Saved MWT", truncated_saved_mwt, truncated_percentage_saved_mwt,
                            help="This indicator displays the Saved MWT value from the projected value of your selected deep-learning model. The value was from the difference between the actual and predicted output of the deep learning model. The arrow up (green) indicates that the selected date has a saved MWT; the arrow down indicates a loss of MWT in that date.")
    st.divider()
    
with bottom_info_left:
    st.subheader('One Day Metric Indicators of MWT', 
               help="These metric indicators display different categories of MWT numerical values based on your selected date and deep learning model. The values of the metric indicators were based on the result of the projection in the line chart above.",
               divider='gray')

    indicator_total_actual_mwt_title, indicator_total_predict_mwt_title, gauge_total_actual_mwt_title, gauge_total_predict_mwt_title, = st.columns([3,3,3,4])
    indicator_total_actual_mwt, indicator_total_predict_mwt, gauge_total_actual_mwt, gauge_total_predict_mwt, = st.columns([3,3,3,4])

    with indicator_total_actual_mwt_title:
        total_mwt_day = ltf.data
    
        st.markdown("""
        <style>
        .big-font3 {
            font-size:18px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font3"><b>Total Actual MegaWatt</b></p>', unsafe_allow_html=True, help="These metric indicators display Total Actual MegaWatt based on your selected hour and date. The value was indicated in decimal places for accurate readings.")
    
    with indicator_total_actual_mwt:
        wch_colour_box = (238,238,228)
        wch_colour_font = (33,19,13)
        fontsize = 25
        valign = "left"
        iconname = "fas fa-asterisk"

        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 0.75); 
                                color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 7px; 
                                padding-left: 12px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height:65px;'>
                                <i class='{iconname} fa-xs'></i><b> {total_mwt_day[min_calendar_val:selected_on_calendar].sum()}</b>
                                </style><BR><span style='font-size: 16px; 
                                margin-top: 0;'>{"MegaWatts"}</style></span></p>"""

        st.markdown(htmlstr, unsafe_allow_html=True)
       
    with indicator_total_predict_mwt_title:
        st.markdown("""
        <style>
        .big-font4 {
            font-size:18px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font4"><b>Total Predicted MegaWatt</b></p>', unsafe_allow_html=True, help="These metric indicators display Total Predicted MegaWatts based on your selected hour and date.")
    with indicator_total_predict_mwt:
        wch_colour_box = (58,59,60)
        wch_colour_font = (238,238,228)
        fontsize = 25
        valign = "left"
        iconname = "fas fa-asterisk"
        
        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 0.75); 
                                color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 7px; 
                                padding-left: 12px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height:65px;'>
                                <i class='{iconname} fa-xs'></i><b> {df_total_predictions[min_calendar_val:selected_on_calendar].sum()} </b> 
                                </style>
                                <BR>
                                <span style='font-size: 16px;'>{"MegaWatts"}</style></span></p>"""

        st.markdown(htmlstr, unsafe_allow_html=True)
        
    with gauge_total_actual_mwt_title:
        st.markdown("""
        <style>
        .big-font1 {
            font-size:18px !important;
            
        }
        </style>
        """, unsafe_allow_html=True)
    
        st.markdown('<p class="big-font1"><b>Actual MWT value</b></p>', unsafe_allow_html=True, help="These metric indicators display the actual MegaWatts based on your selected hour and date. The value was indicated in decimal places for accurate readings.")
    
    with gauge_total_actual_mwt:
        total_mwt_day = MWT_data.loc[:, 'MWT']
        max_val_actual_mw = np.amax(data[min_calendar_val:selected_on_calendar])

        
        wch_colour_box = (29,132,234)
        wch_colour_font = (238,238,228)
        fontsize = 25
        valign = "left"
        iconname = "fas fa-asterisk"
        
        htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                    {wch_colour_box[1]}, 
                                                    {wch_colour_box[2]}, 0.75); 
                                color: rgb({wch_colour_font[0]}, 
                                        {wch_colour_font[1]}, 
                                        {wch_colour_font[2]}, 0.75); 
                                font-size: {fontsize}px; 
                                border-radius: 7px; 
                                padding-left: 12px; 
                                padding-top: 18px; 
                                padding-bottom: 18px; 
                                line-height:65px;'>
                                <i class='{iconname} fa-xs'></i><b> {data[min_calendar_val + (selected_value_hour - 1)]} </b> 
                                </style>
                                <BR>
                                <span style='font-size: 16px;'>{"MegaWatts"}</style></span></p>"""

        st.markdown(htmlstr, unsafe_allow_html=True)
        
    with gauge_total_predict_mwt_title:
        st.markdown("""
        <style>
        .big-font {
            font-size:18px !important;
            
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font"><b>Predicted MWT value</b></p>', unsafe_allow_html=True, help="These metric indicators display Predicted Mega Watts based on your selected hour and date. The color green indicates that the predicted MWT was lower than the actual which means you save an MWT, otherwise, red means the predicted MWT was higher than the actual. The value was indicated in decimal places for accurate readings")
    
    with gauge_total_predict_mwt:
        max_val_predicted_mw = np.amax(df_total_predictions[min_calendar_val:selected_on_calendar])
        if data[min_calendar_val + (selected_value_hour - 1)] >= df_total_predictions[min_calendar_val + (selected_value_hour - 1)]:
            wch_colour_box = (49, 150, 38)
            wch_colour_font = (238,238,228)
            fontsize = 25
            valign = "left"
            iconname = "fas fa-asterisk"
            
            htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                        {wch_colour_box[1]}, 
                                                        {wch_colour_box[2]}, 0.75); 
                                    color: rgb({wch_colour_font[0]}, 
                                            {wch_colour_font[1]}, 
                                            {wch_colour_font[2]}, 0.75); 
                                    font-size: {fontsize}px; 
                                    border-radius: 7px; 
                                    padding-left: 12px; 
                                    padding-top: 18px; 
                                    padding-bottom: 18px; 
                                    line-height:65px;'>
                                    <i class='{iconname} fa-xs'></i><b> {df_total_predictions[min_calendar_val + (selected_value_hour - 1)]} </b> 
                                    </style>
                                    <BR>
                                    <span style='font-size: 16px;'>{"MegaWatts"}</style></span></p>"""

            st.markdown(htmlstr, unsafe_allow_html=True)
        else:
            wch_colour_box = (255, 43, 43)
            wch_colour_font = (238,238,228)
            fontsize = 25
            valign = "left"
            iconname = "fas fa-asterisk"
            
            htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                        {wch_colour_box[1]}, 
                                                        {wch_colour_box[2]}, 0.75); 
                                    color: rgb({wch_colour_font[0]}, 
                                            {wch_colour_font[1]}, 
                                            {wch_colour_font[2]}, 0.75); 
                                    font-size: {fontsize}px; 
                                    border-radius: 7px; 
                                    padding-left: 12px; 
                                    padding-top: 18px; 
                                    padding-bottom: 18px; 
                                    line-height:65px;'>
                                    <i class='{iconname} fa-xs'></i><b> {df_total_predictions[min_calendar_val + (selected_value_hour - 1)]} </b> 
                                    </style>
                                    <BR>
                                    <span style='font-size: 16px;'>{"MegaWatts"}</style></span></p>"""

            st.markdown(htmlstr, unsafe_allow_html=True)