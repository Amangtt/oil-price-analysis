import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wbdata

class Econometric:
    def fetch_data(self,indicator_code, indicator_name, country='WLD', start_date=None, end_date=None):
        """Fetches data from World Bank Data for a specified indicator."""
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=country, date=(start_date, end_date))
        return data

    def clean_data(self,df):
        """Cleans the DataFrame by resetting index, renaming columns, and handling missing values."""
        if df is not None and not df.empty:
            df.reset_index(inplace=True)
            df.columns = ['date', df.columns[1]]  # Keep the first column as 'date', second as the indicator name
            df.dropna(inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return pd.DataFrame() 

    def convert_to_daily(self,df):
        """Converts a DataFrame with dates to a daily frequency."""
        full_index = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        df_daily = df.set_index('date').reindex(full_index)
        df_daily.interpolate(method='time', inplace=True)  # Interpolate to fill missing values
        df_daily.reset_index(inplace=True)
        df_daily.rename(columns={'index': 'Date'}, inplace=True)
        return df_daily

    def Econometric_analysis(self):
        # Set the indicator codes
        gdp_indicator = 'NY.GDP.MKTP.CD'                # GDP (current US$)
        cpi_indicator = 'FP.CPI.TOTL.ZG'                    # Inflation (CPI)
        unemployment_indicator = 'SL.UEM.TOTL.ZS'        # Unemployment rate (% of total labor force)
        exchange_rate_indicator = 'PA.NUS.FCRF'         # Exchange rate, USD to other currencies

        # Define the date range
        start_date = '1987-05-20'
        end_date = '2022-11-14'

        # Fetch and clean data for each indicator
        gdp_data = self.clean_data(self.fetch_data(gdp_indicator, 'GDP', country='WLD', start_date=start_date, end_date=end_date))
        cpi_data = self.clean_data(self.fetch_data(cpi_indicator, 'CPI', country='WLD', start_date=start_date, end_date=end_date))
        unemployment_data = self.clean_data(self.fetch_data(unemployment_indicator, 'Unemployment Rate', country='WLD', start_date=start_date, end_date=end_date))
        exchange_rate_data = self.clean_data(self.fetch_data(exchange_rate_indicator, 'Exchange Rate', country='EMU', start_date=start_date, end_date=end_date))

        # Convert to daily frequency
        gdp_data_daily = self.convert_to_daily(gdp_data)
        cpi_data_daily = self.convert_to_daily(cpi_data)
        unemployment_data_daily = self.convert_to_daily(unemployment_data)
        exchange_rate_data_daily = self.convert_to_daily(exchange_rate_data)


        return gdp_data_daily, cpi_data_daily, unemployment_data_daily, exchange_rate_data_daily

    def analyze_indicators(self,gdp_data, inflation_data, unemployment_data, exchange_rate_data, oil_data):

        
        # Function to merge and analyze
        def analyze_and_plot(indicator_data, indicator_name, oil_data, x_label):
            merged_data = pd.merge(indicator_data, oil_data.reset_index(), on='Date')
            
            # Drop NaN values to ensure correlation calculation is valid
            merged_data.dropna(inplace=True)
            correlation = merged_data[indicator_name].corr(merged_data['Price'])
            print(f"Correlation between {indicator_name} and oil prices: {correlation}")

            # Scatter plot
            plt.figure(figsize=(10, 4))
            sns.scatterplot(data=merged_data, x=indicator_name, y='Price')
            plt.title(f'{indicator_name} vs Brent Oil Prices')
            plt.xlabel(x_label)
            plt.ylabel('Brent Oil Price ($)')
            plt.show()

        # Analyze GDP
        analyze_and_plot(gdp_data, 'GDP', oil_data, 'GDP Growth Rate (%)')

        # Analyze Inflation
        analyze_and_plot(inflation_data, 'CPI', oil_data, 'Inflation Rate (%)')

        # Analyze Unemployment
        analyze_and_plot(unemployment_data, 'Unemployment Rate', oil_data, 'Unemployment Rate (%)')

        # Analyze Exchange Rate
        analyze_and_plot(exchange_rate_data, 'Exchange Rate', oil_data, 'Exchange Rate (USD to Local Currency)')



    def oil_crud(self,oil,crud):
        df=pd.merge(oil,crud,on='year',how='inner')
        c=df[['Price','Crude_Oil_Production']].corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(c,annot=True,fmt='.2f',
        cmap='coolwarm',square=True)
        plt.show()
