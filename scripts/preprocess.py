import pandas as pd

class cleaning:
    def load_data(inputfile):
        df=pd.read_csv(inputfile)
        return df

    def check_missing_values(df):
        return df.isnull().sum()

    def check_duplicate_values(df):
        return df.duplicated().sum()

    def data_types(df):
        return df.dtypes
    def yearly(df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df = df.drop('Date', axis=1)
        agg=df.groupby('year')['Price'].mean().reset_index()
        #df = agg[agg['year'] != 1987]
        agg.to_csv('./Data/oil_price.csv', index=False)
        
  
    def filter_row(df):
        countries_to_keep = ['United States', 'United Kingdom', 'China', 'Russia', 'Japan', 
                         'Germany', 'France', 'India', 'Italy', 'Canada']
        df=df[df['Country Name'].isin(countries_to_keep)]
        melted_df = pd.melt(
        df,
        id_vars=['Country Name', 'Indicator Name'],  # Columns to keep as is
        var_name='year',  # Name for the new year column
        value_name='GDP'  # Name for the new GDP column
    )

        # Step 2: Reshape the DataFrame (make countries columns and years rows)
        pivot_df = melted_df.pivot_table(
            index='year',  # Rows will be years
            columns='Country Name',  # Columns will be countries
            values='GDP'  # Values will be GDP
        )

        # Reset the index to make 'Year' a column instead of an index
        pivot_df = pivot_df.reset_index()

        # Display the final DataFrame
        pivot_df.to_csv('./Data/GDP.csv', index=False)
    
   
    def changes(df):
        df['Date']=pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df = df.drop('Date', axis=1)
        start_date = 1987
        end_date = 2022

        # Filter rows between the two dates
        df = df[(df['year'] >= start_date) & (df['year'] <= end_date)]
        df=df.rename(columns={'U.S. Field Production of Crude Oil (Thousand Barrels per Day)': 'Crude_Oil_Production'})
        agg=df.groupby('year')['Crude_Oil_Production'].mean().reset_index()
       
        agg.to_csv('./Data/crude_oil.csv', index=False)
