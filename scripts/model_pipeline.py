import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
import warnings
from keras.callbacks import EarlyStopping
from arch import arch_model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error


class models:
    def feature_eng(self,merged_data):
        merged_data['Price_Pct_Change'] = merged_data['Price'].pct_change()

        # Rolling averages (e.g., 7-day and 30-day moving averages)
        merged_data['Price_7D_MA'] = merged_data['Price'].rolling(window=7).mean()
        merged_data['Price_30D_MA'] = merged_data['Price'].rolling(window=30).mean()

        # Rolling volatility (e.g., 7-day and 30-day standard deviation)
        merged_data['Price_7D_Volatility'] = merged_data['Price'].rolling(window=7).std()
        merged_data['Price_30D_Volatility'] = merged_data['Price'].rolling(window=30).std()

        # Price momentum (e.g., 7-day and 30-day difference)
        merged_data['Price_7D_Change'] = merged_data['Price'] - merged_data['Price'].shift(7)
        merged_data['Price_30D_Change'] = merged_data['Price'] - merged_data['Price'].shift(30)

        # Lagged Features
        merged_data['Price_7D_Change'] = merged_data['Price'].diff(periods=7)
        merged_data['Price_30D_Change'] = merged_data['Price'].diff(periods=30)
        merged_data.dropna(inplace=True)
        features = merged_data.set_index('Date')
        return features
    #t=feature_eng(merged_data)

    def scale(self,data):
        data.index = pd.to_datetime(data.index)

        # Create a complete date range and reindex
        full_date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(full_date_range)

        # Fill missing values using forward fill
        data.ffill(inplace=True)  # This will fill NaNs with the last valid observation

        # Now you can proceed with your train-validation-test split
        train_size = int(len(data) * 0.8)  # Adjust split ratio for train
        validation_size = int(len(data) * 0.1)  # Adjust split ratio for validation (10% of total)
        # Define train, validation, and test sets
        train = data.iloc[:train_size]
        validation = data.iloc[train_size:train_size + validation_size]
        test = data.iloc[train_size + validation_size:]
        
        return train,validation,test
    

    #VAR model
    def var_model(self,train, test):
        model = VAR(train[['GDP', 'Exchange Rate', 'Price']])
        model_fit = model.fit()

        # Get the number of lags
        lag_order = model_fit.k_ar

        # Prepare the last 'lag_order' observations for forecasting
        last_obs = train[['GDP', 'Exchange Rate', 'Price']].values[-lag_order:]

        # Forecast the future values
        forecast = model_fit.forecast(last_obs, steps=len(test))

        # Create a DataFrame for the predictions
        predictions = pd.DataFrame(forecast, index=test.index, columns=['GDP', 'Exchange Rate', 'Price'])

        # Evaluate the model's performance
        self.evaluate_model(predictions['Price'], test['Price'])

        return predictions

    # Define the evaluation function
    def evaluate_model(self,predictions, actual):
        mse = np.mean((predictions - actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual))

        print(f'MSE: {mse}, RMSE: {rmse}, MAE: {mae}')

        plt.figure(figsize=(14, 7))
        plt.plot(actual.index, actual, label='Actual Price', color='blue')
        plt.plot(predictions.index, predictions, label='Predicted Price', color='orange')
        plt.title('VAR Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
  


    def evaluate_model(self,predictions, actual):
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)

        print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}')

        plt.figure(figsize=(14, 7))
        plt.plot(actual.index, actual, label='Actual Price', color='blue')
        plt.plot(predictions.index, predictions, label='Predicted Price', color='orange')
        plt.title('Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

   
    def sarimax_model(self,train, test):
        # Define endogenous and exogenous variables
        y_train = train['Price']
        exog_train = train[['GDP', 'Exchange Rate', 'Price_Pct_Change',
                            'Price_7D_MA', 'Price_30D_MA',
                            'Price_7D_Volatility', 'Price_30D_Volatility',
                            'Price_7D_Change', 'Price_30D_Change']]

        y_test = test['Price']
        exog_test = test[['GDP', 'Exchange Rate', 'Price_Pct_Change',
                        'Price_7D_MA', 'Price_30D_MA',
                        'Price_7D_Volatility', 'Price_30D_Volatility',
                        'Price_7D_Change', 'Price_30D_Change']]

        try:
            # Fit the SARIMAX model with adjusted parameters
            model = sm.tsa.SARIMAX(y_train,
                                    exog=exog_train,
                                    order=(1, 1, 1),   # Start simple
                                    seasonal_order=(1, 1, 1, 12),  # Start simple for seasonality
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

            results = model.fit(maxiter=1000, disp=False, pgtol=1e-4)  # Adjust convergence settings

            # Forecasting the next values
            forecast = results.get_forecast(steps=len(test), exog=exog_test)
            predictions = forecast.predicted_mean

            # Evaluate the model's performance
            self.evaluate_model(predictions, y_test)

            return predictions

        except Exception as e:
            print(f"An unexpected error occurred: {e}")


  


    def ltsm(self,train,validation,test):
        def prepare_data_lstm(data):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

            # Create sequences
            X, y = [], []
            for i in range(30, len(scaled_data)):
                X.append(scaled_data[i-30:i])
                y.append(scaled_data[i])

            return np.array(X), np.array(y), scaler

        # Assuming train and validation datasets are defined
        X_train, y_train, scaler = prepare_data_lstm(train)
        X_val, y_val, _ = prepare_data_lstm(validation)  # Prepare validation data

        # Build LSTM Model using Input Layer
        def create_lstm_model():
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Input(shape=(30, 1)))  # 30 time steps and 1 feature
            model.add(tf.keras.layers.LSTM(50, return_sequences=True))
            model.add(tf.keras.layers.LSTM(50))
            model.add(tf.keras.layers.Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model_lstm = create_lstm_model()

        # Define early stopping to protect overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=5,          # Number of epochs with no improvement after which training will be stopped
            verbose=1,           # Verbosity mode
            mode='min',          # Mode can be 'min' for loss
            restore_best_weights=True  # Restore the best weights from training
        )
        # Train the LSTM Model with validation data
        history = model_lstm.fit(X_train, y_train, epochs=50, batch_size=32,
                                validation_data=(X_val, y_val), callbacks=[early_stopping])  # Include validation data

        # Prepare test data
        X_test, y_test, _ = prepare_data_lstm(test)

        # Predictions
        predictions_lstm = model_lstm.predict(X_test)
        predictions_lstm = scaler.inverse_transform(predictions_lstm)

        # Calculate evaluation metrics
        mse = mean_squared_error(test['Price'][30:], predictions_lstm)
        mae = mean_absolute_error(test['Price'][30:], predictions_lstm)

        print(f'MSE: {mse:.2f}, MAE: {mae:.2f}')

        # Visualizing LSTM Predictions
        plt.figure(figsize=(14, 7))
        plt.plot(test.index, test['Price'], label='Actual Price', color='blue')
        plt.plot(test.index[30:], predictions_lstm, label='Predicted Price (LSTM)', color='orange')
        plt.title('LSTM Model Predictions vs Actual Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
  
