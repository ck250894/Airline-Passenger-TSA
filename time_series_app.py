# Loading required packages - 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ HOME PAGE

def home_page():
    st.write("""
             # Time Series Analysis
             
             This is our TSA app.
             """)
    
    st.write(" ")
    
    about_expander = st.beta_expander("About")
    
    about_expander.write("""
                         Time Series Analysis!  
             
                         This app serves two purposes -  
                         * **Australia Airline Traffic:** This page shows the analysis done by the author on tweets of NBA teams, to assist the marketing team. They can examine the sentiment of the tweets and the overall trends/polarity of the tweets. This can assist them in making an informed decision about where to allocate their spend.
                         * **User Area:** Users can use this area to upload their own data or a set of data and do simple text processing with the help of a simple UI.
                         
                         """)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS PAGE

# FUNCTIONS - 


def get_city_data(data, text_column):
    if text_column != "~ All Cities ~":
        return data[data['City2'] == text_column]
    else:
        return data

def plot_boxplots(df, text_column, box_col1, box_col2):
    total_passengers = df.groupby([df.index]).agg({"Passenger_Trips" : "sum"})
    
    total_passengers['year'] = [d.year for d in total_passengers.index]
    total_passengers['month'] = [d.strftime('%b') for d in total_passengers.index]
    # years = total_passengers['year'].unique()
    # Plotting Year wise
    temp = total_passengers["2000":"2020"]
    fig, axes = plt.subplots(1, 1, figsize=(10,7), dpi= 80)
    sns.boxplot(x='year', y='Passenger_Trips', data=temp,
                ax=axes)
    axes.set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
    
    box_col1.pyplot()
    
    # Plotting Monthly
    temp = total_passengers["2016":"2019"]
    fig, axes = plt.subplots(1, 1, figsize=(10,7), dpi= 80)
    sns.boxplot(x='month', y='Passenger_Trips', data=temp)
    axes.set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
    box_col2.pyplot()

def get_forecast(df, forecast_col2):
    
    df = df.groupby([df.index]).agg({"Passenger_Trips" : "sum"})
    df = df.dropna()
    
    model = SARIMAX(np.log(df["Passenger_Trips"]), order = (2,1,1), seasonal_order = (0, 1, 1, 12))
    results = model.fit()
    
    fcast = np.exp(results.get_forecast('2021-09').summary_frame())
    fig, ax = plt.subplots(figsize=(15, 5))
    df.loc['2015':]['Passenger_Trips'].plot(ax=ax)
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)
    forecast_col2.pyplot()
    
    plt.plot(df[df.index.to_series().between('2019-01-01', '2019-12-01')]['Passenger_Trips'], color='green')
    plt.plot(np.exp(results.predict(start='2019-01-01', end='2019-12-01')), color='red')
    plt.title('Comparison of Actual and Predicted data')
    plt.xticks(rotation=90)
    forecast_col2.pyplot()


def airline_page():
    st.title("Airline Analysis Page")
    
    st.write("""
             EXPLAIN!  
             
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    
    st.write("")
    about_expander = st.beta_expander("About")
    about_expander. write("""
                          Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
                          """)
    st.write("")
    
    # Plotting the graphs for yearly and monthly - 
    # Initial overview - 
    st.write("""
             ## Initial Overview of Airline Data 
             
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             
             """)
    
    image_col1, image_col2, image_col3 = st.beta_columns((1,2,1))
    
    # ~~~~~ GRAPH
    # Setting the image - 
    image = Image.open('Data_Analysis/smooth_monthly.png')
    
    image_col2.image(image, use_column_width=True)
    
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    
    st.write(" ")
    
    st.write("""
             ### Now since we have an idea of the airline trend, the user can further explore below. The selectbox can be used to explore data from different cities.
             """)
    
    
    st.write(" ")
    col1, col2, col3 = st.beta_columns((1,3,1))
    
    
    city_option = ["~ All Cities ~"]
    df_cities = list(data["City2"].unique())
    col_options = city_option + df_cities
    text_column = col2.selectbox('Choose the city you want to explore', col_options, index = 0)
    col2.write(" ")
    
    st.write("""
             * Below we can see 2 graphs. The first one one is yearly ... And second is monthly variations in the past 5 years.
             """)
    st.write(" ")
    # Get city data - 
    temp = get_city_data(data, text_column)
    
    box_col1, box_col2 = st.beta_columns((1,1))
    plot_boxplots(temp, text_column, box_col1, box_col2)
    st.write(" ")
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    st.write(" ")
    # Showing the best model according to all cities - 
    forecast_col1, forecast_col2, forecast_col3 = st.beta_columns((1,1.5, 1))
    get_forecast(temp, forecast_col2)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ USER PAGE

def user_page():
    st.title("User Basic Page")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN PAGE

# Setting the page layout -
st.set_page_config(layout = 'wide', page_title = "TSA App")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Loading in the data - 
data = pd.read_csv("dom_citypairs_web.csv", parse_dates = [[10, 11]], infer_datetime_format = True, index_col = 0)

# Choosing the desired columns - 
data = data[["City1", "City2", "Passenger_Trips"]]
data["City2"] = data["City2"].apply(lambda x: x.title())


# Setting the image - 
image = Image.open('airline3.png')
if image.mode != "RGB":
    image = image.convert('RGB')
# Setting the image width -
st.image(image, use_column_width=True)

navigation_option = st.sidebar.selectbox("Navigation Tab", ['Home Page', 'Airline Page', 'Basic TSA'])

if navigation_option == "Home Page":
    home_page()
elif navigation_option == "Airline Page":
    airline_page()
elif navigation_option == "Basic TSA":
    user_page()
                             
