import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

# Caching Function for zd and zd_melt df
@st.cache_data
def load_data():
    zd = pd.read_csv("data/zhvi_bdrmcnt_labeled.csv")
    zd_melt = pd.melt(
        zd,
        id_vars=['Zip', 'State', 'CountyName', 'BedroomCount', 'population', 'density', 'city'],
        var_name='date',
        value_name='ZHVI'
    )
    zd_melt['date'] = pd.to_datetime(zd_melt['date'])
    return zd, zd_melt

def plot(zd, zd_melt, zip_code, bedroom_count, zip_city_state_str):
    # Melt ZD and Datetime Data
    st.markdown(f"## ARIMA Model Forecast for: <span style='color:blue'>{bedroom_count} Bedroom in {zip_city_state_str}</span>", unsafe_allow_html=True)

    # Selection
    city_data = zd_melt[(zd_melt['Zip'] == zip_code) & (zd_melt['BedroomCount'] == bedroom_count)]
    city_data.set_index('date', inplace=True)
    city_data.sort_index(inplace=True)
    
    # Ensure the frequency is set
    city_data = city_data.asfreq('ME')

    # Train Test Split -- last 12 months as test
    ts = city_data['ZHVI']
    train = ts[:-12]
    test = ts[-12:]
    
    # Fit the best ARIMA model - Forecast future values -- 12 on test and 36 for 3 years after
    # Fig 1 - No testing
    model1 = ARIMA(ts, order=(5, 1, 1))
    model_fit1 = model1.fit()
    forecast1 = model_fit1.forecast(steps=48)
    forecast_index1 = pd.date_range(start=test.index[0], periods=49, freq='ME')[1:]
    forecast_series1 = pd.Series(forecast1, index=forecast_index1)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ts.index, y=ts, mode='lines', name='Train'))
    fig1.add_trace(go.Scatter(x=forecast_series1.index, y=forecast_series1, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))

    fig1.update_layout(
        title=f'{zip_code} - ZHVI Forecast',
        xaxis_title='Date',
        yaxis_title='ZHVI ($)',
        width=1000,
        height=500
    )
    
    # Fig 2 - Train Test Forecast
    model2 = ARIMA(train, order=(5, 1, 1))
    model_fit2 = model2.fit()
    forecast2 = model_fit2.forecast(steps=48)
    forecast_index2 = pd.date_range(start=test.index[0], periods=49, freq='ME')[1:]
    forecast_series2 = pd.Series(forecast2, index=forecast_index2)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train, mode='lines', name='Train', visible='legendonly'))
    fig2.add_trace(go.Scatter(x=test.index, y=test, mode='lines', name='Test', line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=forecast_series2.index, y=forecast_series2, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))
    fig2.update_layout(
        title=f'{zip_code} - ZHVI Forecast - With Test Set (July 2023 - July 2024)',
        xaxis_title='Date',
        yaxis_title='ZHVI ($)',
        width=1000,
        height=500
    )

    # Display Figs in 2 Column
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig1)
    with col2:
        st.plotly_chart(fig2)

    
    # Performance measures
    aic = model_fit2.aic
    bic = model_fit2.bic
    hqic = model_fit2.hqic
    log_likelihood = model_fit2.llf
    forecast_values = forecast2.values[:12]
    test_values = test.values


    # Error Measures
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(forecast_values - test_values))

    # Mean Squared Error (MSE)
    mse = np.mean((forecast_values - test_values) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    epsilon = 1e-10
    mape = np.mean(np.abs((forecast_values - test_values) / (test_values + epsilon))) * 100

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = 100 * np.mean(np.abs(forecast_values - test_values) / (np.abs(forecast_values) + np.abs(test_values) + epsilon) / 2)

    # R-squared (R^2)
    ss_total = np.sum((test_values - np.mean(test_values)) ** 2)
    ss_residual = np.sum((test_values - forecast_values) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

        
    # Display Performance Metrics
    st.markdown("##### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
                <div style="color: blue; font-weight: bold;">AIC</div>
                {aic:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
                <div style="color: blue; font-weight: bold;">BIC</div>
                {bic:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
                <div style="color: blue; font-weight: bold;">HQIC</div>
                {hqic:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
                <div style="color: blue; font-weight: bold;">Log Likelihood</div>
                {log_likelihood:.2f}
            </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    # Display Error Metrics
    st.markdown("##### Error Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">MAE</div>
                {mae:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">MSE</div>
                {mse:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">RMSE</div>
                {rmse:.2f}
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">MAPE</div>
                {mape:.2f}%
            </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">SMAPE</div>
                {smape:.2f}%
            </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
                <div style="color: red; font-weight: bold;">R²</div>
                {r_squared:.2f}
            </div>
        """, unsafe_allow_html=True)


def display_sidebar(zd):
    # Zip City State Selection
    st.sidebar.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)
    st.sidebar.write(f"## City Selection")
    
    zip_city_state = zd[['Zip', 'city', 'State']]
    zip_city_state['Selection'] = zip_city_state.apply(lambda row: f"{row['Zip']} - {row['city']}, {row['State']}", axis=1)
    selected_zip_city_state = st.sidebar.selectbox('Select ZIP code, city, and state', zip_city_state['Selection'])
    
    # Bedroom Count selection
    selected_row = zip_city_state[zip_city_state['Selection'] == selected_zip_city_state].iloc[0]
    zip_code = selected_row['Zip']
    city = selected_row['city']
    state = selected_row['State']
    
    bedroom_count = zd[zd['Zip'] == zip_code]['BedroomCount']
    selected_bedroom_count = st.sidebar.selectbox('Select Bedroom Count', sorted(bedroom_count))
    
    return zip_code, city, state, selected_bedroom_count

def display_selected_data(zd, zip_code, city, state, selected_bedroom_count):
    # Display Selection and Population and Density
    st.sidebar.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)
    st.sidebar.write(f"## Selected Data")
    st.sidebar.write(f"**ZIP Code**: {zip_code}")
    st.sidebar.write(f"**City**: {city}")
    st.sidebar.write(f"**State**: {state}")
    if selected_bedroom_count == 5:
        st.sidebar.write(f"**Bedroom Count**: 5 or more")
    else:
        st.sidebar.write(f"**Bedroom Count**: {selected_bedroom_count}")

    filtered_data = zd[(zd['Zip'] == zip_code) & (zd['BedroomCount'] == selected_bedroom_count)]
    
    if not filtered_data.empty:
        st.sidebar.write(f"**Population**: {filtered_data['population'].values[0]}")
        st.sidebar.write(f"**Density**: {filtered_data['density'].values[0]}")
    else:
        st.sidebar.write("No data available for the selected criteria.")

def display_about_zhvi():
    st.sidebar.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)
    st.sidebar.markdown("""
        ## Zillow Home Value Index (ZHVI)

        The **Zillow Home Value Index (ZHVI)** is a measure of the typical home value and market changes across a given region and housing type. It reflects the typical value for homes in the 35th to 65th percentile range. 
                      
        [Link to more info](https://www.zillow.com/research/data/)
    """)

def main():
    # Page Config
    st.set_page_config(
        page_title="ZHVI Forecaster",
        page_icon="./images/zhvi_forecaster_logo_house.png"
    )
    
    # Styles
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 100%;
        padding-left: 1%;
        padding-right: 1%;
        top: 0px;
        max-height:100%;
        padding-top: 1%;
        padding-bottom: 1%;        
    }
    .sidebar-title {
        color: blue;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.image("./images/zhvi_forecaster_logo_full.png", use_column_width=True)
    
    # Load data
    zd, zd_melt = load_data()
    
    # Display sidebar content
    zip_code, city, state, selected_bedroom_count = display_sidebar(zd)
    
    # Display selected data
    display_selected_data(zd, zip_code, city, state, selected_bedroom_count)
    display_about_zhvi()

    # Zip City State String for title
    zip_city_state_str = f"{zip_code} - {city}, {state}"
    
    # Plot
    if zip_code and selected_bedroom_count:
        with st.spinner('Generating plot...'):
            plot(zd, zd_melt, zip_code, selected_bedroom_count, zip_city_state_str)
    else:
        st.write("Please make selections to see the plot.")

if __name__ == "__main__":
    main()