from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from analysis import (
    load_price_data,
    calculate_price_trends, 
    calculate_yearly_average_price, 
    calculate_analysis_metrics,calculate_price_distribution, 
    calculate_event_impact, get_prices_around_event
)

app = Flask(__name__)
CORS(app)

# Load data
price_data = load_price_data()

key_events = {
    'OPEC Cuts & Asian Economic Recovery':'1999-03-23',
    'Geopolitical Tensions':'2004-06-23',
    'Hurricanes & Supply Disruptions':'2005-08-29',
    'Rising Demand & Speculation':'2007-11-01',
    'Financial Crisis':'2008-07-11',
    'Oil Price Crash':'2008-11-18',
    'Post-Crisis Recovery':'2009-06-11',
    'Global Economic Recovery & BP Oil Spill':'2010-12-13',
    'ISIS insurgency in Iraq':'2014-06-19',
    'Oil Price Crash(2014)':'2014-11-28',
    'Continued Oversupply':'2015-01-12',
    'OPEC Production Cuts':'2017-01-03',
    'OPEC Production Cuts':'2017-01-03',
    'Russia-Saudi Arabia oil price war':'2020-03-09',
    'COVID-19 Pandemic':'2020-04-21',
    'OPEC+ production limits':'2021-07-19',
    'Russia-Ukraine War':'2022-03-01',
}

@app.route('/api/price-trends', methods=['GET'])
def get_price_trends():
    trends_data = []
    for event, date in key_events.items():
        event_date = pd.to_datetime(date)
        prices_around_event = get_prices_around_event(event_date, price_data, days_before=180, days_after=180)
        trends_data.append({
            'event': event,
            'date': date,
            'prices': prices_around_event['Price'].tolist(),
            'dates': prices_around_event.index.tolist()
        })
    return jsonify(trends_data)

@app.route('/api/event-impact', methods=['GET'])
def get_event_impact():
    results = []
    for event, date in key_events.items():
        impact_data = calculate_event_impact(event, date, price_data)
        results.append(impact_data)
    return jsonify(results)

@app.route('/api/analysis-metrics', methods=['GET'])
def get_analysis():
    try:
        analysis_results = calculate_analysis_metrics(price_data.reset_index())
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error

@app.route('/api/prices', methods=['GET'])
def get_price_trend():
    try:
        # Generate full dataset without filtering
        price_data_dict = calculate_price_trends(price_data)
        
        return jsonify(price_data_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/average-yearly-price', methods=['GET'])
def get_yearly_average():
    try:
        analysis_results = calculate_yearly_average_price(price_data)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error
    
@app.route('/api/price-distribution', methods=['GET'])
def get_distribution():
    try:
        analysis_results = calculate_price_distribution(price_data)
        return jsonify(analysis_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Internal Server Error    
if __name__ == '__main__':
    app.run(debug=True,port=5001)