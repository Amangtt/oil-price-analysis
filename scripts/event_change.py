import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import pymc as pm
import arviz as az
from datetime import timedelta
from scipy import stats

class EventChangeAnalyzer:


    def __init__(self, price_data, logger=None):
        self.price_data = price_data
        self.mean_price = self.price_data['Price'].mean()
        
        
    def calculate_cusum(self):
        """Calculates and plots the CUSUM of deviations from the mean price."""
    
        cusum = (self.price_data['Price'] - self.mean_price).cumsum()
        plt.plot(self.price_data.index, cusum, label='CUSUM of Price Deviations', color='orange')
        plt.title('CUSUM Line Plot of Brent Oil Price Deviations')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sum of Deviations (USD)')
        plt.legend()
        plt.grid()
        plt.show()
            
      

    def detect_change_point(self, n_bkps=5):
        """Detects change points using the CUSUM-based method from the ruptures package."""
        try:
            
            # Extract the price series for change point detection
            price_df = self.price_data.reset_index()
            price_series = price_df['Price'].values

            # Apply the CUSUM-based method for change point detection
            model = 'rbf'
            algo = rpt.Pelt(model=model).fit(price_series)
            change_points = algo.predict(pen=20)  # Adjust n_bkps for more or fewer breakpoints

            # Extract and print the year of each change point
            change_years = [price_df['Date'].iloc[cp].year for cp in change_points[:-1]]  # Exclude the last index (end of data)
            print("Detected change point years:", change_years)

            # Plotting the Brent Oil Price with change points
            plt.plot(price_df['Date'], price_df['Price'], label='Brent Oil Price', color='blue')

            # Overlay detected change points with year annotations
            for cp in change_points[:-1]:  # Exclude the last point (end of data)
                year = price_df['Date'].iloc[cp].year
                plt.axvline(price_df['Date'].iloc[cp], color='red', linestyle='--')
                plt.text(price_df['Date'].iloc[cp], price_df['Price'].iloc[cp], str(year), color="red", fontsize=10)

            # Enhancements
            plt.title('Brent Oil Prices with CUSUM Change Points and Years')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid()
            plt.show()
          
        except Exception as e:
            print("Error detecting change points: %s", e)
            
            
    def bayesian_change_point_detection(self):
        """Performs Bayesian change point analysis using PyMC."""
        try:
            data = self.price_data['Price'].values
            prior_mu = np.mean(data)
            
            with pm.Model() as model:
                change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(data) - 1)
                mu1 = pm.Normal('mu1', mu=prior_mu, sigma=5)
                mu2 = pm.Normal('mu2', mu=prior_mu, sigma=5)
                sigma1 = pm.HalfNormal('sigma1', sigma=5)
                sigma2 = pm.HalfNormal('sigma2', sigma=5)
                
                likelihood = pm.Normal(
                    'likelihood',
                    mu=pm.math.switch(change_point >= np.arange(len(data)), mu1, mu2),
                    sigma=pm.math.switch(change_point >= np.arange(len(data)), sigma1, sigma2),
                    observed=data
                )
                
                trace = pm.sample(4000, tune=2000, chains=4, random_seed=42)
                print("Bayesian sampling completed successfully.")
                
                az.plot_trace(trace)
                plt.show()
                
                s_posterior = trace.posterior['change_point'].values.flatten()
                change_point_estimate = int(np.median(s_posterior))
                change_point_date = self.price_data.index[change_point_estimate]
                
                print(f"Estimated Change Point Date: {change_point_date}")
                
                
                return change_point_date
        except Exception as e:
            print("Error in Bayesian change point analysis: %s", e)
    

    def _get_prices_around_event(self, event_date, days_before=30, days_after=30):
        """Helper function to get prices around a given event date."""
        before_date = event_date - timedelta(days=days_before)
        after_date = event_date + timedelta(days=days_after)
        prices_around_event = self.price_data[(self.price_data.index >= before_date) & (self.price_data.index <= after_date)]
        return prices_around_event

    def analyze_price_changes_around_events(self, key_events):
        """Analyzes and plots price changes around specific events."""
        results = []

        for event, date in key_events.items():
            try:
                event_date = pd.to_datetime(date)
                prices_around_event = self._get_prices_around_event(event_date, days_before=180, days_after=180)

                # Calculate percentage changes at different intervals
                change_1m = self._calculate_percentage_change(event_date, 30)
                change_3m = self._calculate_percentage_change(event_date, 90)
                change_6m = self._calculate_percentage_change(event_date, 180)

                # Calculate cumulative returns around the event
                cum_return_before = prices_around_event.loc[:event_date].pct_change().add(1).cumprod().iloc[-1] - 1
                cum_return_after = prices_around_event.loc[event_date:].pct_change().add(1).cumprod().iloc[-1] - 1

                results.append({
                    "Event": event,
                    "Date": date,
                    "Change_1M": change_1m,
                    "Change_3M": change_3m,
                    "Change_6M": change_6m,
                    "Cumulative Return Before": cum_return_before['Price'],
                    "Cumulative Return After": cum_return_after['Price']
                })
            except KeyError:
                print("Event %s at %s is out of price data range.", event, date)

        event_impact_df = pd.DataFrame(results)
        self._plot_price_trends_around_events(key_events)
        self._plot_percentage_changes_and_cumulative_returns(event_impact_df)
        t_test_df = self._perform_statistical_analysis(key_events)

        return event_impact_df, t_test_df

    def _calculate_percentage_change(self, event_date, days):
        """Calculates the percentage change in price before and after a given number of days around an event."""
        try:
            price_before = self.price_data.loc[event_date - timedelta(days=days), 'Price']
            price_after = self.price_data.loc[event_date + timedelta(days=days), 'Price']
            return ((price_after - price_before) / price_before) * 100
        except KeyError:
            return None

    def _plot_price_trends_around_events(self, key_events, days_before=180, days_after=180):
        """Plots price trends around specified events."""
        plt.figure(figsize=(14, 8))
        for event, date in key_events.items():
            event_date = pd.to_datetime(date)
            prices_around_event = self._get_prices_around_event(event_date, days_before=days_before, days_after=days_after)
            plt.plot(prices_around_event.index, prices_around_event['Price'], label=f"{event} ({date})")
            plt.axvline(event_date, color='red', linestyle='--', linewidth=0.8)
        
        plt.title("Brent Oil Price Trends Around Key Events")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def _plot_percentage_changes_and_cumulative_returns(self, event_impact_df):
        """Plots percentage changes and cumulative returns before and after events."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot for percentage changes
        sns.barplot(data=event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Change_1M", "Change_3M", "Change_6M"]),
                    x="Event", y="value", hue="variable", ax=axes[0])
        axes[0].set_title("Percentage Change in Brent Oil Prices Before and After Events")
        axes[0].set_ylabel("Percentage Change")
        axes[0].legend(title="Change Period")

        # Plot for cumulative returns
        sns.barplot(data=event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Cumulative Return Before", "Cumulative Return After"]),
                    x="Event", y="value", hue="variable", ax=axes[1])
        axes[1].set_title("Cumulative Returns Before and After Events")
        axes[1].set_ylabel("Cumulative Return")
        axes[1].legend(title="Cumulative Return")

        plt.tight_layout()
        plt.show()

    def _perform_statistical_analysis(self, key_events):
        """Performs a t-test to assess significant price changes before and after events."""
        t_test_results = {}
        for event, date in key_events.items():
            event_date = pd.to_datetime(date)
            try:
                before_prices = self._get_prices_around_event(event_date, days_before=180).loc[:event_date]['Price']
                after_prices = self._get_prices_around_event(event_date, days_after=180).loc[event_date:]['Price']
                t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')
                t_test_results[event] = {"t-statistic": t_stat, "p-value": p_val}
            except KeyError:
                print("Event %s at %s is out of price data range.", event, date)

        t_test_df = pd.DataFrame(t_test_results).T
        print(t_test_df)
        return t_test_df