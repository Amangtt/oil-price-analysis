import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def trend_per_year(df):
        df['Date']=pd.to_datetime(df['Date'])
        df['year']=df['Date'].dt.year
        df['Weekday'] = df['Date'].dt.day_name()
        yearly_trend = df.groupby('year')['Price'].mean().reset_index()
        # Plotting the trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_trend, x='year', y='Price', marker='o')
        plt.title('Brent Oil Per barrel Price Trend Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Average Price (USD)')
        plt.grid(True)
        plt.show()

  

    #trend per month
    def trend_per_month(df):
        df['Date']=pd.to_datetime(df['Date'])
        df['month']=df['Date'].dt.month
        yearly_trend = df.groupby('month')['Price'].mean().reset_index()
        # Plotting the trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_trend, x='month', y='Price', marker='o')
        plt.title('Brent Oil Per barrel Price Trend Over the months')
        plt.xlabel('month')
        plt.ylabel('Average Price (USD)')
        plt.grid(True)
        plt.show()


    #trend per month
    def trend_per_day(df):
        df['Date']=pd.to_datetime(df['Date'])
        df['day']=df['Date'].dt.day
        yearly_trend = df.groupby('day')['Price'].mean().reset_index()
        # Plotting the trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=yearly_trend, x='day', y='Price', marker='o')
        plt.title('Brent Oil Per barrel Price Trend Over the days')
        plt.xlabel('day')
        plt.ylabel('Average Price (USD)')
        plt.grid(True)
        plt.show()
   

    def detect_outliers_with_boxplot(df):
            
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Price'], color="skyblue")
        plt.title(f"Box Plot of {'Price'}", fontsize=16)
        plt.xlabel('Price', fontsize=14)
        plt.show()
 

    def lay(df):
        plt.figure(figsize=(15, 5))
        sns.histplot(df['Price'], bins=10, kde=True)
        plt.title(f'Histogram of {'Price'}')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
