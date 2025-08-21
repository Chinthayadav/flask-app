import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, jsonify, request, make_response, send_file
import requests
import json
import time
import threading
import os
import finnhub  # Keep finnhub for the dashboard part if desired
import pandas as pd
import yfinance as yf  # Import yfinance for DRIP calculator data
from datetime import datetime, timedelta, timezone  # Import timezone

# For PDF generation
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

app = Flask(__name__)

# --- Configuration for Finnhub (Dashboard) ---
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "d1nlrb1r01qovv8k2q6gd1nlrb1r01qovv8k2q70")
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Default stock symbols for the dashboard display
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "IBM", "META", "JPM", "KO", "PG", "UNH", "VOO", "SPY",
                 "INTC", "PEP", "V", "MA", "DIS", "NFLX", "ADBE", "CRM", "ORCL", "CSCO", "BA", "WMT", "CVX", "XOM",
                 "BAC", "T", "NKE", "MCD", "HD", "PFE", "MRK", "ABT", "TMO", "LLY", "COST", "AVGO", "GE", "DHR",
                 "BMY", "CAT", "QCOM", "AMAT", "AMD", "FDX", "UPS", "GILD", "AXP", "DE", "BKNG", "ZTS"]
FETCH_INTERVAL_SECONDS = 30  # How often to refresh dashboard data
latest_stock_data = {}  # Stores the latest stock data for the dashboard
data_lock = threading.Lock()  # Lock for thread-safe access to latest_stock_data

def fetch_and_update_dashboard_stock_data(symbol, api_key):
    """
    Fetches real-time stock data for the dashboard using Finnhub API.
    Updates the global latest_stock_data dictionary.
    """
    quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
    company_profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={api_key}"

    try:
        quote_response = requests.get(quote_url)
        quote_response.raise_for_status()
        quote_data = quote_response.json()

        profile_response = requests.get(company_profile_url)
        profile_response.raise_for_status()
        profile_data = profile_response.json()

        if not quote_data or quote_data.get('c') is None:
            print(f"No valid quote data for {symbol}")
            return

        company_name = profile_data.get('name', symbol)
        
        current_price = quote_data.get('c')
        high_price = quote_data.get('h')
        low_price = quote_data.get('l')
        open_price = quote_data.get('o')
        prev_close_price = quote_data.get('pc')

        change = current_price - prev_close_price
        percentage_change = (change / prev_close_price * 100) if prev_close_price else 0

        with data_lock:
            latest_stock_data[symbol] = {
                "symbol": symbol,
                "company_name": company_name,
                "logo": profile_data.get('logo', ''),
                "current_price": f"{current_price:.2f}",
                "high_price": f"{high_price:.2f}",
                "low_price": f"{low_price:.2f}",
                "open_price": f"{open_price:.2f}",
                "prev_close_price": f"{prev_close_price:.2f}",
                "change": f"{change:.2f}",
                "percentage_change": f"{percentage_change:.2f}",
                "timestamp": int(time.time() * 1000)
            }
            print(f"Updated dashboard data for {symbol} ({company_name})")

    except requests.exceptions.RequestException as req_err:
        print(f"Network or API error fetching dashboard data for {symbol}: {req_err}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decoding error for {symbol}: {json_err}")
    except Exception as e:
        print(f"An unexpected error occurred fetching dashboard data for {symbol}: {e}")

def background_data_updater():
    """
    Background thread function to periodically fetch and update stock data for the dashboard.
    """
    while True:
        start_time = time.time()
        for symbol in STOCK_SYMBOLS:
            fetch_and_update_dashboard_stock_data(symbol, FINNHUB_API_KEY)
            time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        time_to_sleep = FETCH_INTERVAL_SECONDS - elapsed_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

updater_thread = threading.Thread(target=background_data_updater, daemon=True)
updater_thread.start()

# --- Helper Function for DRIP Calculator Data Fetching (using yfinance) ---
def get_drip_stock_data(ticker_symbol):
    """
    Fetches comprehensive stock and dividend data for DRIP calculation using yfinance.
    Returns a dictionary with relevant data or an error dictionary if an error occurs.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        if not info:
            return {"error": f"No information found for ticker: {ticker_symbol}. It might be an invalid symbol or data is unavailable from yfinance."}

        current_price = info.get('currentPrice')
        if current_price is None:
            return {"error": f"Current price data not available for {ticker_symbol}. Data might be delayed or unavailable from yfinance."}

        long_name = info.get('longName') or ticker_symbol

        dividend_yield = info.get('dividendYield')
        if dividend_yield is None:
            dividend_yield = info.get('forwardAnnualDividendYield')
        if dividend_yield is None and info.get('dividendRate') and current_price:
            dividend_yield = info['dividendRate'] / current_price
        
        annual_dividend_rate_per_share = info.get('dividendRate')

        end_date = datetime.now(timezone.utc)
        start_date_5yr = end_date - timedelta(days=5 * 365)
        
        historical_dividends_series = ticker.dividends.loc[start_date_5yr:end_date]
        
        annual_dividends_by_year = {}
        for date, dividend_amount in historical_dividends_series.items():
            year = date.year
            annual_dividends_by_year[year] = annual_dividends_by_year.get(year, 0) + dividend_amount
        
        sorted_years = sorted(annual_dividends_by_year.keys())
        
        dividend_growth_rate = 0.0
        if len(sorted_years) >= 2:
            first_year_div = annual_dividends_by_year.get(sorted_years[0], 0)
            last_year_div = annual_dividends_by_year.get(sorted_years[-1], 0)
            num_years = sorted_years[-1] - sorted_years[0]
            if first_year_div > 0 and num_years > 0:
                dividend_growth_rate = ((last_year_div / first_year_div) ** (1 / num_years)) - 1
            elif last_year_div > 0 and first_year_div == 0:
                dividend_growth_rate = 0.0
        
        hist_prices_df = ticker.history(period="5y")
        
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "ticker": ticker_symbol,
            "longName": long_name,
            "currentPrice": current_price,
            "dividendYield": round(dividend_yield * 100, 2) if dividend_yield is not None else None,
            "annualDividendRate": round(annual_dividend_rate_per_share, 2) if annual_dividend_rate_per_share is not None else None,
            "annualDividendGrowthRate": round(min(dividend_growth_rate * 100, 5.0), 2) if dividend_growth_rate is not None else None,
            "payoutFrequency": "N/A",
            "historicalDividendsSeries": historical_dividends_series,
            "annualDividendsByYear": annual_dividends_by_year,
            "historicalPrices": hist_prices_df,
            "lastUpdated": last_updated
        }
    except Exception as e:
        print(f"Error fetching DRIP data for {ticker_symbol}: {e}")
        return {"error": f"Failed to fetch data for {ticker_symbol}. Reason: {e}. Please check the symbol or try again later.", "lastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# --- NEW API Route for getting stock price ---
@app.route('/api/get_stock_price/<ticker>')
def get_stock_price(ticker):
    """
    API endpoint to get current stock price for a given ticker symbol.
    Used for real-time calculation of initial shares.
    """
    try:
        ticker_symbol = ticker.upper().strip()
        stock_data = get_drip_stock_data(ticker_symbol)
        
        if "error" in stock_data:
            return jsonify({"error": stock_data["error"]}), 400
            
        return jsonify({
            "ticker": ticker_symbol,
            "currentPrice": stock_data["currentPrice"],
            "longName": stock_data["longName"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stock price for {ticker}: {str(e)}"}), 500

# --- Flask Routes ---
@app.route('/')
def dashboard():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/api/stocks')
def api_stocks():
    """Provides the latest stock data for the dashboard as JSON."""
    with data_lock:
        return jsonify(latest_stock_data)
        
@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    """Renders the dividend calculator page (GET) and handles the dividend calculation (POST)."""
    results = []
    comparison_table = None
    
    def limit_to_7_digits(value):
        """Limit a number to 7 digits by rounding to the nearest 7-digit value and capping growth."""
        if value >= 1000000:  # Cap at 1 million to ensure realism
            return round(value, -6) // 1000 * 1000
        return round(value, 2)

    if request.method == 'POST':
        initial_investment = request.form.get('initial_investment', type=float)
        investment_years = request.form.get('investment_years', type=int)
        drip_enabled = request.form.get('drip_enabled')
        selected_tickers = request.form.getlist('tickers')
        initial_share_price = request.form.get('initial_share_price', type=float)
        dividend_yield = request.form.get('dividend_yield', type=float)
        annual_dividend_growth_rate = request.form.get('annual_dividend_growth_rate', type=float)
        annual_stock_price_growth_rate = request.form.get('annual_stock_price_growth_rate', type=float)
        payout_frequency = request.form.get('payout_frequency')
        new_initial_share_price = request.form.get('new_initial_share_price', type=float)

        # Validate inputs
        if not initial_investment or initial_investment <= 0:
            results.append({'error': 'Please provide a positive initial investment.'})
        if not investment_years or investment_years <= 0:
            results.append({'error': 'Please provide a positive investment duration.'})
        if not selected_tickers:
            results.append({'error': 'Please select at least one ticker symbol.'})
        if not initial_share_price or initial_share_price <= 0:
            results.append({'error': 'Please provide a positive initial share price.'})
        if dividend_yield is not None and dividend_yield < 0:
            results.append({'error': 'Please provide a non-negative dividend yield.'})
        if annual_dividend_growth_rate is not None and annual_dividend_growth_rate < 0:
            results.append({'error': 'Please provide a non-negative annual dividend growth rate.'})
        if annual_stock_price_growth_rate is not None and annual_stock_price_growth_rate < 0:
            results.append({'error': 'Please provide a non-negative annual stock price growth rate.'})
        if not payout_frequency:
            results.append({'error': 'Please select a dividend payout frequency.'})

        if not results:
            comparison_data_for_table = []
            
            for ticker_symbol in selected_tickers:
                ticker_symbol = ticker_symbol.strip().upper()
                if not ticker_symbol:
                    continue

                drip_data = get_drip_stock_data(ticker_symbol)
                if "error" in drip_data:
                    results.append({'ticker': ticker_symbol, 'error': drip_data['error'], 'last_updated': drip_data.get('lastUpdated', 'N/A')})
                    continue
                
                current_price_calc = drip_data['currentPrice'] if drip_data['currentPrice'] is not None else initial_share_price
                div_yield_decimal_calc = (drip_data['dividendYield'] / 100) if drip_data['dividendYield'] is not None else (dividend_yield / 100 if dividend_yield is not None else 0.0)
                div_growth_decimal_calc = (drip_data['annualDividendGrowthRate'] / 100) if drip_data['annualDividendGrowthRate'] is not None else (annual_dividend_growth_rate / 100 if annual_dividend_growth_rate is not None else 0.0)
                stock_price_growth_decimal_calc = (annual_stock_price_growth_rate / 100) if annual_stock_price_growth_rate is not None else 0.0
                
                annual_dividend_per_share_start = drip_data['annualDividendRate'] if drip_data['annualDividendRate'] is not None else (current_price_calc * div_yield_decimal_calc)
                
                if annual_dividend_per_share_start == 0:
                    results.append({
                        'ticker': ticker_symbol,
                        'current_price': f"${limit_to_7_digits(current_price_calc):,.2f}",
                        'error': f"{ticker_symbol} does not currently pay dividends or dividend yield is zero.",
                        'last_updated': drip_data.get('lastUpdated', 'N/A')
                    })
                    continue

                current_shares_drip = initial_investment / current_price_calc if current_price_calc > 0 else 0
                current_shares_no_drip = initial_investment / current_price_calc if current_price_calc > 0 else 0
                current_stock_price = current_price_calc
                annual_dividend_per_share = annual_dividend_per_share_start
                total_dividends_received_no_drip = 0
                yearly_breakdown_for_display = []
                portfolio_values_with_drip_chart = [{'x': 0, 'y': limit_to_7_digits(initial_investment)}]
                shares_owned_with_drip_chart = [{'x': 0, 'y': limit_to_7_digits(current_shares_drip)}]
                years_for_chart = [0]

                number_of_payouts = {'Annual': 1, 'Semi-Annual': 2, 'Quarterly': 4, 'Monthly': 12}.get(payout_frequency, 4)
                total_periods = investment_years * number_of_payouts

                for period in range(total_periods):
                    period_fraction = period / number_of_payouts
                    year = int(period // number_of_payouts) + 1

                    stock_price_at_period_start = current_price_calc * (1 + stock_price_growth_decimal_calc) ** period_fraction
                    dividend_per_share_period = (annual_dividend_per_share_start / number_of_payouts) * (1 + div_growth_decimal_calc) ** period_fraction

                    dividends_this_period_drip = current_shares_drip * dividend_per_share_period
                    dividends_this_period_no_drip = current_shares_no_drip * dividend_per_share_period

                    if drip_enabled == 'yes' and stock_price_at_period_start > 0:
                        shares_bought = dividends_this_period_drip / stock_price_at_period_start
                        current_shares_drip += shares_bought
                    else:
                        total_dividends_received_no_drip += dividends_this_period_no_drip

                    if (period + 1) % number_of_payouts == 0:
                        year_end = year
                        portfolio_value_drip = current_shares_drip * stock_price_at_period_start
                        portfolio_value_no_drip = current_shares_no_drip * stock_price_at_period_start + total_dividends_received_no_drip

                        yearly_dividends = sum(
                            current_shares_drip * (annual_dividend_per_share_start / number_of_payouts) * 
                            (1 + div_growth_decimal_calc) ** ((p + period - number_of_payouts + 1) / number_of_payouts)
                            for p in range(number_of_payouts)
                        ) if year_end > 1 else dividends_this_period_drip * number_of_payouts

                        reinvested_shares_year = current_shares_drip - (float(yearly_breakdown_for_display[-1]['Shares Owned (Start)'].replace(',', '')) if yearly_breakdown_for_display else current_shares_drip)

                        yearly_breakdown_for_display.append({
                            'Year': year_end,
                            'Shares Owned (Start)': f"{limit_to_7_digits(current_shares_drip):,.2f}",
                            'Stock Price (Start)': f"${limit_to_7_digits(stock_price_at_period_start):,.2f}",
                            'Annual Dividend Per Share': f"${limit_to_7_digits(dividend_per_share_period * number_of_payouts):,.2f}",
                            'Dividends Received (Year)': f"${limit_to_7_digits(yearly_dividends):,.2f}",
                            'Reinvested Shares (Year)': f"{limit_to_7_digits(reinvested_shares_year):,.2f}",
                            'Shares Owned (End)': f"{limit_to_7_digits(current_shares_drip):,.2f}",
                            'Portfolio Value (End)': f"${limit_to_7_digits(portfolio_value_drip):,.2f}"
                        })

                        portfolio_values_with_drip_chart.append({'x': year_end, 'y': limit_to_7_digits(portfolio_value_drip)})
                        shares_owned_with_drip_chart.append({'x': year_end, 'y': limit_to_7_digits(current_shares_drip)})
                        years_for_chart.append(year_end)

                final_stock_price = current_price_calc * (1 + stock_price_growth_decimal_calc) ** investment_years
                final_dividend_per_share = annual_dividend_per_share_start * (1 + div_growth_decimal_calc) ** investment_years
                final_value_with_drip = current_shares_drip * final_stock_price
                final_value_no_drip = current_shares_no_drip * final_stock_price + total_dividends_received_no_drip
                final_shares_no_drip = current_shares_no_drip

                yield_on_cost = (final_dividend_per_share / initial_share_price * 100) if initial_share_price > 0 else 0.0

                historical_dividend_chart_data = []
                if drip_data['historicalDividendsSeries'] is not None and not drip_data['historicalDividendsSeries'].empty:
                    annual_div_plot_series = drip_data['historicalDividendsSeries'].resample('Y').sum()
                    for date, amount in annual_div_plot_series.items():
                        historical_dividend_chart_data.append({'x': date.strftime('%Y-%m-%d'), 'y': limit_to_7_digits(amount)})

                historical_price_chart_data = []
                if drip_data['historicalPrices'] is not None and not drip_data['historicalPrices'].empty:
                    for date, price in drip_data['historicalPrices']['Close'].items():
                        historical_price_chart_data.append({'x': date.strftime('%Y-%m-%d'), 'y': limit_to_7_digits(price)})

                results.append({
                    'ticker': ticker_symbol,
                    'current_price': f"${limit_to_7_digits(current_price_calc):,.2f}",
                    'annual_dividend_per_share': f"${limit_to_7_digits(annual_dividend_per_share_start):,.2f}",
                    'dividend_yield': f"{limit_to_7_digits(drip_data['dividendYield'] if drip_data['dividendYield'] is not None else (dividend_yield if dividend_yield is not None else 0)):,.2f}%",
                    'payout_ratio': f"{drip_data.get('payoutRatioTTM', 'N/A')}",
                    'initial_shares': f"{limit_to_7_digits(initial_investment / current_price_calc):,.2f}",
                    'final_value_no_drip': f"${limit_to_7_digits(final_value_no_drip):,.2f}",
                    'final_shares_no_drip': f"{limit_to_7_digits(final_shares_no_drip):,.2f}",
                    'final_value_with_drip': f"${limit_to_7_digits(final_value_with_drip):,.2f}",
                    'final_shares_with_drip': f"{limit_to_7_digits(current_shares_drip):,.2f}",
                    'yearly_breakdown_display': yearly_breakdown_for_display,
                    'historical_dividend_chart_data': historical_dividend_chart_data,
                    'historical_price_chart_data': historical_price_chart_data,
                    'drip_portfolio_value_chart_data': portfolio_values_with_drip_chart,
                    'drip_shares_owned_chart_data': shares_owned_with_drip_chart,
                    'payout_frequency_fetched': payout_frequency,
                    'last_updated': drip_data.get('lastUpdated', 'N/A'),
                    'yield_on_cost': f"{limit_to_7_digits(yield_on_cost):,.2f}%",
                    'error': None
                })

                comparison_data_for_table.append({
                    'Ticker': ticker_symbol,
                    'Initial Investment': f"${limit_to_7_digits(initial_investment):,.2f}",
                    'Initial Shares': f"{limit_to_7_digits(initial_investment / current_price_calc):,.2f}",
                    'Current Price': f"${limit_to_7_digits(current_price_calc):,.2f}",
                    'Annual Div per Share': f"${limit_to_7_digits(annual_dividend_per_share_start):,.2f}",
                    'Dividend Yield': f"{limit_to_7_digits(drip_data['dividendYield'] if drip_data['dividendYield'] is not None else (dividend_yield if dividend_yield is not None else 0)):,.2f}%",
                    'Payout Frequency': payout_frequency,
                    '5-Yr Div Growth': f"{limit_to_7_digits(drip_data['annualDividendGrowthRate'] if drip_data['annualDividendGrowthRate'] is not None else (annual_dividend_growth_rate if annual_dividend_growth_rate is not None else 0)):,.2f}%",
                    'Final Value (No DRIP)': f"${limit_to_7_digits(final_value_no_drip):,.2f}",
                    'Final Value (With DRIP)': f"${limit_to_7_digits(final_value_with_drip):,.2f}"
                })
                
                time.sleep(0.1)

            if len(selected_tickers) > 1 and not any(r.get('error') for r in results):
                if comparison_data_for_table:
                    comparison_data_df = pd.DataFrame(comparison_data_for_table)
                    comparison_table = comparison_data_df.to_html(classes='table table-striped table-bordered mt-3', index=False)
            
    return render_template('calculator.html', 
                          results=results, 
                          comparison_table=comparison_table,
                          stock_symbols=STOCK_SYMBOLS,
                          selected_tickers=selected_tickers if request.method == 'POST' else [],
                          initial_investment=None,
                          investment_years=None,
                          initial_share_price=initial_share_price if request.method == 'POST' else None,
                          dividend_yield=None,
                          annual_dividend_growth_rate=None,
                          annual_stock_price_growth_rate=None,
                          payout_frequency=None,
                          drip_enabled=None,
                          new_initial_share_price=None)

@app.route('/export_csv', methods=['POST'])
def export_csv():
    """Exports the detailed dividend calculation results to a CSV file."""
    tickers_str = request.form.get('export_tickers_hidden')
    initial_investment = request.form.get('export_initial_investment', type=float)
    investment_years = request.form.get('export_investment_years', type=int)
    drip_enabled = request.form.get('export_drip_enabled') == 'yes'
    initial_share_price_export = request.form.get('export_initial_share_price', type=float)
    dividend_yield_export = request.form.get('export_dividend_yield', type=float)
    annual_dividend_growth_rate_export = request.form.get('export_annual_dividend_growth_rate', type=float)
    annual_stock_price_growth_rate_export = request.form.get('export_annual_stock_price_growth_rate', type=float)
    payout_frequency_export = request.form.get('export_payout_frequency','Quarterly')

    def limit_to_7_digits(value):
        """Limit a number to 7 digits by rounding to the nearest 7-digit value and capping growth."""
        if value >= 1000000:  # Cap at 1 million to ensure realism
            return round(value, -6) // 1000 * 1000
        return round(value, 2)

    if not tickers_str or not initial_investment or not investment_years:
        return "Missing data for export. Please ensure all required fields are present.", 400

    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
    all_data_for_export = []

    for ticker_symbol in tickers:
        drip_data = get_drip_stock_data(ticker_symbol)
        if "error" in drip_data:
            all_data_for_export.append({'Ticker': ticker_symbol, 'Error': drip_data['error']})
            continue

        current_price_calc = drip_data['currentPrice'] if drip_data.get('currentPrice') is not None else initial_share_price_export
        div_yield_decimal_calc = (drip_data['dividendYield'] / 100) if drip_data.get('dividendYield') is not None else (dividend_yield_export / 100 if dividend_yield_export is not None else 0.0)
        div_growth_decimal_calc = (drip_data['annualDividendGrowthRate'] / 100) if drip_data.get('annualDividendGrowthRate') is not None else (annual_dividend_growth_rate_export / 100 if annual_dividend_growth_rate_export is not None else 0.0)
        stock_price_growth_decimal_calc = (annual_stock_price_growth_rate_export / 100) if annual_stock_price_growth_rate_export is not None else 0.0
        payout_frequency_sim = payout_frequency_export
        
        annual_dividend_per_share_start = drip_data['annualDividendRate'] if drip_data['annualDividendRate'] is not None else (current_price_calc * div_yield_decimal_calc)

        if annual_dividend_per_share_start == 0 and not drip_enabled:
            all_data_for_export.append({'Ticker': ticker_symbol, 'Error': f"{ticker_symbol} does not pay dividends or dividend yield is zero."})
            continue

        current_shares = initial_investment / current_price_calc if current_price_calc > 0 else 0
        if current_shares == 0:
            all_data_for_export.append({'Ticker': ticker_symbol, 'Error': f"Initial share price is zero or invalid, cannot calculate. Please check inputs."})
            continue

        current_stock_price = current_price_calc
        annual_dividend_per_share = annual_dividend_per_share_start
        
        yearly_breakdown = []
        yearly_breakdown.append([
            'Year', 'Shares Owned (Start)', 'Stock Price (Start)', 
            'Annual Div Per Share', 'Dividends Received (Year)', 
            'Reinvested Shares (Year)', 'Shares Owned (End)', 'Portfolio Value (End)'
        ])

        yearly_breakdown.append([
            '0',
            f"{limit_to_7_digits(current_shares):,.2f}",
            f"${limit_to_7_digits(current_price_calc):,.2f}",
            f"${limit_to_7_digits(annual_dividend_per_share_start):,.2f}",
            "$0.00",
            "0.00",
            f"{limit_to_7_digits(current_shares):,.2f}",
            f"${limit_to_7_digits(initial_investment):,.2f}"
        ])

        for year in range(1, investment_years + 1):
            shares_at_start_of_year = current_shares
            stock_price_at_start_of_year = current_stock_price
            annual_dividend_per_share_at_start_of_year = annual_dividend_per_share

            dividends_received_this_year = 0
            reinvested_shares_this_year = 0

            number_of_payouts = {'Annual': 1, 'Semi-Annual': 2, 'Quarterly': 4, 'Monthly': 12}.get(payout_frequency_sim.capitalize(), 1)
            period_stock_price_growth_rate = (1 + stock_price_growth_decimal_calc) ** (1/number_of_payouts) - 1
            period_dividend_growth_rate = (1 + div_growth_decimal_calc) ** (1/number_of_payouts) - 1

            if drip_enabled:
                for i in range(number_of_payouts):
                    current_stock_price_for_reinvestment = current_stock_price * ((1 + period_stock_price_growth_rate) ** i)
                    current_dividend_per_share_for_period = (annual_dividend_per_share_at_start_of_year / number_of_payouts) * ((1 + period_dividend_growth_rate) ** i)

                    dividends_received_this_period = shares_at_start_of_year * current_dividend_per_share_for_period
                    
                    shares_bought_this_period = 0
                    if current_stock_price_for_reinvestment > 0:
                        shares_bought_this_period = dividends_received_this_period / current_stock_price_for_reinvestment
                    
                    current_shares += shares_bought_this_period
                    dividends_received_this_year += dividends_received_this_period
                    reinvested_shares_this_year += shares_bought_this_period
            else:
                dividends_received_this_year = shares_at_start_of_year * annual_dividend_per_share_at_start_of_year
                reinvested_shares_this_year = 0

            current_stock_price = current_stock_price * (1 + stock_price_growth_decimal_calc)
            annual_dividend_per_share = annual_dividend_per_share_at_start_of_year * (1 + div_growth_decimal_calc)

            portfolio_value_end_of_year = current_shares * current_stock_price

            yearly_breakdown.append([
                str(year),
                f"{limit_to_7_digits(shares_at_start_of_year):,.2f}",
                f"${limit_to_7_digits(stock_price_at_start_of_year):,.2f}",
                f"${limit_to_7_digits(annual_dividend_per_share_at_start_of_year):,.2f}",
                f"${limit_to_7_digits(dividends_received_this_year):,.2f}",
                f"{limit_to_7_digits(reinvested_shares_this_year):,.2f}",
                f"{limit_to_7_digits(current_shares):,.2f}",
                f"${limit_to_7_digits(portfolio_value_end_of_year):,.2f}"
            ])
            
        all_data_for_export.extend(yearly_breakdown)
        time.sleep(0.1)

    df = pd.DataFrame(all_data_for_export)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, float_format="%.2f")
    csv_buffer.seek(0)

    response = make_response(csv_buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=dividend_calculator_results.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    """Exports the detailed dividend calculation results to a PDF file."""
    try:
        # --- 1. Corrected Data Retrieval from Form (using _pdf suffixes) ---
        tickers_str = request.form.get('export_tickers_hidden_pdf')
        initial_investment = request.form.get('export_initial_investment_pdf', type=float)
        investment_years = request.form.get('export_investment_years_pdf', type=int)
        drip_enabled = request.form.get('export_drip_enabled_pdf') == 'yes'
        
        initial_share_price_export = request.form.get('export_initial_share_price_pdf', type=float)
        dividend_yield_export = request.form.get('export_dividend_yield_pdf', type=float)
        annual_dividend_growth_rate_export = request.form.get('export_annual_dividend_growth_rate_pdf', type=float)
        annual_stock_price_growth_rate_export = request.form.get('export_annual_stock_price_growth_rate_pdf', type=float)
        payout_frequency_export = request.form.get('export_payout_frequency_pdf', 'Quarterly')

        def limit_to_7_digits(value):
            """Limit a number to 7 digits by rounding to the nearest 7-digit value."""
            if value >= 10000000 or value <= -10000000:
                return round(value, -7 + len(str(int(abs(value))))) // 10000 * 10000
            return round(value, 2)

        if not tickers_str or initial_investment is None or investment_years is None:
            return "Missing data for export. Please ensure all required fields are present.", 400

        tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        styles = getSampleStyleSheet()
        story = []

        # --- Summary Information (consistent with previous PDF) ---
        story.append(Paragraph("Dividend Reinvestment Plan (DRIP) Calculator Results", styles['h1']))
        story.append(Spacer(1, 0.2 * 2.54 * 72)) 

        summary_text = f"<b>Initial Investment:</b> ${limit_to_7_digits(initial_investment):,.2f}<br/>" \
                       f"<b>Investment Duration:</b> {investment_years} Years<br/>" \
                       f"<b>DRIP Enabled:</b> {'Yes' if drip_enabled else 'No'}<br/>" \
                       f"<b>Default Initial Share Price (if API fails):</b> ${limit_to_7_digits(initial_share_price_export):,.2f}<br/>" \
                       f"<b>Default Dividend Yield (if API fails):</b> {limit_to_7_digits(dividend_yield_export):,.2f}%<br/>" \
                       f"<b>Default Annual Dividend Growth Rate (if API fails):</b> {limit_to_7_digits(annual_dividend_growth_rate_export):,.2f}%<br/>" \
                       f"<b>Default Annual Stock Price Growth Rate:</b> {limit_to_7_digits(annual_stock_price_growth_rate_export):,.2f}%<br/>" \
                       f"<b>Default Payout Frequency (if API fails):</b> {payout_frequency_export}"
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2 * 2.54 * 72))

        # --- Loop through tickers, similar to CSV ---
        for ticker_symbol in tickers:
            drip_data = get_drip_stock_data(ticker_symbol)
            if "error" in drip_data:
                story.append(Paragraph(f"Error for {ticker_symbol}: {drip_data['error']}", styles['h3']))
                story.append(Spacer(1, 0.1 * 2.54 * 72))
                continue

            # Use fetched data or export form data if fetched is None
            current_price_calc = drip_data['currentPrice'] if drip_data.get('currentPrice') is not None else initial_share_price_export
            div_yield_decimal_calc = (drip_data['dividendYield'] / 100) if drip_data.get('dividendYield') is not None else (dividend_yield_export / 100)
            div_growth_decimal_calc = (drip_data['annualDividendGrowthRate'] / 100) if drip_data.get('annualDividendGrowthRate') is not None else (annual_dividend_growth_rate_export / 100)
            stock_price_growth_decimal_calc = (annual_stock_price_growth_rate_export / 100)
            # IMPORTANT: Use the payout_frequency_export from the form for calculations
            payout_frequency_sim = payout_frequency_export 
            
            annual_dividend_per_share_start = current_price_calc * div_yield_decimal_calc

            if annual_dividend_per_share_start == 0 and not drip_enabled:
                story.append(Paragraph(f"<b>{ticker_symbol}:</b> Does not pay dividends or dividend yield is zero.", styles['h3']))
                story.append(Spacer(1, 0.1 * 2.54 * 72))
            
            # --- Simulation Initialization ---
            current_shares = initial_investment / current_price_calc if current_price_calc > 0 else 0
            if current_shares == 0:
                story.append(Paragraph(f"<b>{ticker_symbol}:</b> Initial share price is zero or invalid, cannot calculate. Please check inputs.", styles['h3']))
                story.append(Spacer(1, 0.1 * 2.54 * 72))
                continue

            current_stock_price = current_price_calc
            annual_dividend_per_share = annual_dividend_per_share_start
            
            yearly_breakdown_data = []
            # Table Headers
            yearly_breakdown_data.append([
                'Year', 'Shares Owned (Start)', 'Stock Price (Start)', 
                'Annual Div Per Share', 'Dividends Received (Year)', 
                'Portfolio Value (End)'
            ])

            # Initial state (Year 0) for PDF table - Added for clear starting point
            yearly_breakdown_data.append([
                '0',
                f"{limit_to_7_digits(current_shares):,.2f}",
                f"${limit_to_7_digits(current_price_calc):,.2f}",
                f"${limit_to_7_digits(annual_dividend_per_share_start):,.2f}",
                "$0.00",
                "0.00",
                f"{limit_to_7_digits(current_shares):,.2f}",
                f"${limit_to_7_digits(initial_investment):,.2f}"
            ])

            # --- Simulation Loop (consistent with CSV logic) ---
            for year in range(1, investment_years + 1):
                shares_at_start_of_year = current_shares
                stock_price_at_start_of_year = current_stock_price
                annual_dividend_per_share_at_start_of_year = annual_dividend_per_share

                dividends_received_this_year = 0
                reinvested_shares_this_year = 0

                number_of_payouts = {
                    'Annual': 1,
                    'Semi-Annual': 2,
                    'Quarterly': 4,
                    'Monthly': 12
                }.get(payout_frequency_sim.capitalize(), 1)

                # Calculate period growth rates for PDF as well
                period_stock_price_growth_rate = (1 + stock_price_growth_decimal_calc) ** (1/number_of_payouts) - 1
                period_dividend_growth_rate = (1 + div_growth_decimal_calc) ** (1/number_of_payouts) - 1

                if drip_enabled:
                    for i in range(number_of_payouts):
                        # Update stock price and dividend per share for the current period
                        current_stock_price_for_reinvestment = current_stock_price * ((1 + period_stock_price_growth_rate) ** i)
                        current_dividend_per_share_for_period = (annual_dividend_per_share_at_start_of_year / number_of_payouts) * ((1 + period_dividend_growth_rate) ** i)

                        dividends_received_this_period = shares_at_start_of_year * current_dividend_per_share_for_period
                        
                        shares_bought_this_period = 0
                        if current_stock_price_for_reinvestment > 0:
                            shares_bought_this_period = dividends_received_this_period / current_stock_price_for_reinvestment
                        
                        current_shares += shares_bought_this_period
                        dividends_received_this_year += dividends_received_this_period
                        reinvested_shares_this_year += shares_bought_this_period
                else:
                    dividends_received_this_year = shares_at_start_of_year * annual_dividend_per_share_at_start_of_year
                    reinvested_shares_this_year = 0

                # Update stock price and annual dividend per share for the *end* of the year
                current_stock_price = current_stock_price * (1 + stock_price_growth_decimal_calc)
                annual_dividend_per_share = annual_dividend_per_share_at_start_of_year * (1 + div_growth_decimal_calc)

                portfolio_value_end_of_year = current_shares * current_stock_price

                yearly_breakdown_data.append([
                    str(year),
                    f"{limit_to_7_digits(shares_at_start_of_year):,.2f}",
                    f"${limit_to_7_digits(stock_price_at_start_of_year):,.2f}",
                    f"${limit_to_7_digits(annual_dividend_per_share_at_start_of_year):,.2f}",
                    f"${limit_to_7_digits(dividends_received_this_year):,.2f}",
                    f"{limit_to_7_digits(reinvested_shares_this_year):,.2f}",
                    f"{limit_to_7_digits(current_shares):,.2f}",
                    f"${limit_to_7_digits(portfolio_value_end_of_year):,.2f}"
                ])
                
            # Add ticker-specific title and table
            story.append(Paragraph(f"Results for {ticker_symbol}", styles['h2']))
            story.append(Spacer(1, 0.1 * 2.54 * 72))

            # Add yearly breakdown table with colWidths for better layout
            num_columns = len(yearly_breakdown_data[0])
            page_width = landscape(letter)[0] - (doc.leftMargin + doc.rightMargin)
            col_widths = [page_width / num_columns] * num_columns

            table = Table(yearly_breakdown_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a69bd')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f2f5')),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('LEFTPADDING', (0,0), (-1,-1), 4),
                ('RIGHTPADDING', (0,0), (-1,-1), 4),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('FONTSIZE', (0,0), (-1,-1), 7),
            ]))
            story.append(table)
            story.append(Spacer(1, 0.5 * 2.54 * 72))  # Space after each ticker's table
            
            time.sleep(0.1)  # Small delay for API friendliness, consistent with CSV

        doc.build(story)
        buffer.seek(0)
        
        return send_file(buffer, as_attachment=True, download_name='dividend_calculator_results.pdf', mimetype='application/pdf')

    except Exception as e:
        import traceback
        app.logger.error(f"Error in export_pdf: {e}\n{traceback.format_exc()}")
        return f"An error occurred while generating the PDF: {str(e)}. Please check the server logs for more details.", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
