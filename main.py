import os
import datetime
import json
import re
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import nltk
import logging
import time
import string

# Install NLTK corpora
nltk.download('punkt')
nltk.download('stopwords')

# Logging setup
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch the company's name from the stock ticker
def get_company_name(ticker):
    try:
        stock_info = yf.Ticker(ticker).info
        return stock_info.get("longName", ticker)
    except Exception as e:
        print(f"Error fetching company name for ticker {ticker}: {e}")
        return ticker

def get_primary_name(company_name):
    suffixes = ["Inc.", "Incorporated", "Corporation", "Corp.", "Ltd.", "Limited", "LLC", "PLC"]
    words = company_name.split()
    return " ".join([word for word in words if word not in suffixes])


# --------------------------------------------
# CONFIGURATION
# --------------------------------------------
TICKER = "AAPL"  # Example ticker
START_DATE_NEWS = "2010-10-01"
END_DATE_NEWS = "2025-01-01"
LOOKAHEAD_DAYS = 5
# Folder containing WARC files
warc_folder = r"C:\Users\yoelarie\Downloads\warc.paths\downloaded_warc_files"

COMPANY_NAME = get_company_name(TICKER)
PRIMARY_NAME = get_primary_name(COMPANY_NAME)
KEYWORDS = [TICKER, COMPANY_NAME, PRIMARY_NAME]  # Include ticker, full name, and primary name

# --------------------------------------------
# STEP 1: FETCH NEWS ARTICLES FROM ALL WARC FILES
# --------------------------------------------
def save_articles_to_file(articles, filename):
    """Save the articles list (a list of dictionaries) to a JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(articles)} articles to {filename}.")
    except Exception as e:
        print(f"Error saving articles to file {filename}: {e}")

def load_articles_from_file(filename):
    """Load the articles from a JSON file and return them as a list."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            articles = json.load(f)
        print(f"Loaded {len(articles)} articles from {filename}.")
        return articles
    except Exception as e:
        print(f"Error loading articles from file {filename}: {e}")
        return []

def extract_main_text(html_content):
    """
    Extracts the main text content from HTML using BeautifulSoup.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        article_text = " ".join(p.get_text() for p in soup.find_all("p"))
        return article_text if article_text else None
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return None

def extract_snippet(text, keyword, window=20):
    """
    Extracts a snippet of text around the first occurrence of 'keyword',
    returning a substring with up to `window` characters before and after.
    """
    # Use regex to extract a substring around the keyword.
    pattern = rf".{{0,{window}}}{re.escape(keyword)}.{{0,{window}}}"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else ""


def fetch_news_articles(warc_folder, keywords, max_records=1000):
    """
    Process all WARC files in the specified folder and extract relevant articles.

    Parameters:
        warc_folder (str): Path to the folder containing WARC files.
        keywords (list): List of keywords to search in the articles.
        max_records (int): Maximum number of articles to process.

    Returns:
        list: Extracted articles matching the given keywords.
    """
    articles = []
    record_count = 0

    # Identify all subfolders containing "file.warc"
    warc_files = sorted([
        os.path.join(warc_folder, subfolder, subfolder)
        for subfolder in os.listdir(warc_folder)
        if os.path.isdir(os.path.join(warc_folder, subfolder)) and
           os.path.exists(os.path.join(warc_folder, subfolder, subfolder))
    ])

    print(f"Found {len(warc_files)} WARC files in folder {warc_folder}.")

    # List of financial keywords to further filter the articles
    financial_keywords = [
        'earnings', 'revenue', 'profit', 'loss', 'quarter', 'guidance',
        'dividend', 'financial', 'market', 'investment', 'share', 'report',
        'merger', 'acquisition', 'growth', 'forecast', 'outlook', 'valuation',
        'buyback', 'stock', 'bonds', 'interest', 'capital', 'liquidity',
        'volatility', 'inflation', 'deflation', 'debt', 'credit', 'rating',
        'assets', 'liabilities', 'equity', 'leverage', 'bankruptcy', 'cash flow',
        'earnings call', 'EPS', 'net income', 'gross margin', 'operating margin',
        'cost-cutting', 'expenses', 'fiscal', 'monetary', 'diversification',
        'securities', 'derivatives', 'commodities', 'funding', 'reserves'
    ]

    # Iterate over all WARC files
    for file_index, warc_file in enumerate(warc_files):
        print(f"\n[{file_index + 1}/{len(warc_files)}] Processing file: {warc_file}")
        try:
            with open(warc_file, 'rb') as stream:
                total_records = sum(1 for _ in ArchiveIterator(stream))  # Get total records for progress tracking
                stream.seek(0)  # Reset stream position
                print(f"Total records in file: {total_records}")

                for i, record in enumerate(ArchiveIterator(stream)):
                    progress = (i + 1) / total_records * 100
                    if i % 10000 == 0:  # Print every 10,000 records
                        print(f"Processing record {i + 1}/{total_records} ({progress:.2f}%)")

                    if record_count >= max_records:
                        print("Reached max record limit. Stopping extraction.")
                        return articles  # Stop processing further

                    if record.rec_type != 'response':
                        continue

                    try:
                        url = record.rec_headers.get_header("WARC-Target-URI")
                        content = record.content_stream().read()
                        html_text = content.decode("utf-8", errors="ignore")

                        # Extract main text
                        main_text = extract_main_text(html_text)
                        if not main_text:
                            continue
                        main_text = main_text[:1500]  # Limit text size

                        publish_date = None

                        # Attempt to extract publish date from metadata
                        soup = BeautifulSoup(html_text, "lxml")
                        meta_date = soup.find("meta", {"property": "article:published_time"})
                        if meta_date and meta_date.get("content"):
                            publish_date = meta_date["content"][:10]  # Extract YYYY-MM-DD

                        # Fallback: Use WARC record date if available
                        if not publish_date:
                            warc_date = record.rec_headers.get_header("WARC-Date")
                            if warc_date:
                                publish_date = warc_date[:10]  # Extract YYYY-MM-DD

                        # Fallback: Attempt to parse date from the URL (if possible)
                        if not publish_date:
                            date_match = re.search(r'\d{4}-\d{2}-\d{2}', url)
                            if date_match:
                                publish_date = date_match.group(0)

                        # Check if the article contains one of the main stock name keywords
                        if any(keyword.lower() in main_text.lower() for keyword in keywords):

                            # 1. Find the first financial keyword in the article.
                            found_financial_kw = None
                            for fin_kw in financial_keywords:
                                if fin_kw in main_text.lower():
                                    found_financial_kw = fin_kw
                                    break

                            if not found_financial_kw:
                                print(f"Skipping article (no financial keyword found) - {url}")
                                continue  # Skip if no financial keyword is found

                            # 2. Extract a snippet around the found financial keyword.
                            fin_snippet = extract_snippet(main_text, found_financial_kw)

                            # 3. Find the first stock name keyword in the article.
                            found_stock_kw = None
                            stock_snippet = ""
                            for stock_kw in keywords:
                                if stock_kw.lower() in main_text.lower():
                                    found_stock_kw = stock_kw
                                    stock_snippet = extract_snippet(main_text, stock_kw)
                                    if stock_snippet:
                                        break

                            # Skip the article if one of the snippets is empty.
                            if not fin_snippet.strip() or not stock_snippet.strip():
                                print(f"Skipping article (empty snippet) - {url}")
                                continue

                            # Append the article with both snippets.
                            articles.append({
                                'url': url,
                                'text': main_text[:1000],  # Limit to 1000 characters for preview
                                'snippet_financial': fin_snippet,
                                'financial_keyword': found_financial_kw,
                                'snippet_stock': stock_snippet,
                                'stock_keyword': found_stock_kw,
                                'publish_date': publish_date,
                            })
                            record_count += 1
                            print(f"Matched article {record_count}: {url}")
                            print(f"Publish Date: {publish_date}")
                            print(f"Financial Keyword Found: {found_financial_kw}")
                            print(f"Stock Keyword Found: {found_stock_kw}")

                    except Exception as e:
                        logging.error(f"Error processing WARC record: {e}")
                        print(f"Error processing record: {e}")

        except FileNotFoundError:
            logging.error(f"WARC file not found: {warc_file}")
            print(f"Error: WARC file not found - {warc_file}")
        except Exception as e:
            logging.error(f"Unexpected error while processing {warc_file}: {e}")
            print(f"Unexpected error: {e}")

        print(f"Finished processing {warc_file}. Found {len(articles)} matching articles so far.")

    print(f"\nFinished processing all WARC files. Found {len(articles)} matching articles in total.")
    return articles


# --------------------------------------------
# STEP 2: PERFORM SENTIMENT ANALYSIS
# --------------------------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    print("polarity: ", polarity)
    if polarity > 0.05:
        return "positive"
    elif polarity < -0.05:
        return "negative"
    else:
        return "neutral"

# --------------------------------------------
# STEP 3: GET HISTORICAL STOCK DATA
# --------------------------------------------
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if 'Adj Close' in data.columns:
        required_columns.append('Adj Close')
    return data[required_columns]

# --------------------------------------------
# STEP 4: EVALUATE STRATEGY OUTCOMES
# --------------------------------------------
def evaluate_trades(news_articles, stock_data, lookahead_days=5):
    """
    Evaluate trading outcomes based on sentiment and historical stock data.
    Only articles with positive or negative sentiment are kept.

    Parameters:
        news_articles (list): List of articles with sentiment analysis.
        stock_data (pandas.DataFrame): Stock data with historical prices.
        lookahead_days (int): Days to look ahead for evaluation.

    Returns:
        list: Evaluation results.
    """
    results = []
    stock_trading_dates = stock_data.index

    for article in news_articles:
        try:
            # Extract publish date from the article (validate if it exists)
            publish_date = article.get("publish_date")
            if publish_date:
                news_date = datetime.datetime.strptime(publish_date, "%Y-%m-%d").date()
            else:
                print(f"Skipping article with missing or invalid publish date: {article['url']}")
                continue

            # Find the next trading day (day after the news)
            day_after = next((d for d in stock_trading_dates if d.date() > news_date), None)
            if not day_after:
                print(f"No trading data available after {news_date}. Skipping article: {article['url']}")
                continue

            # Perform sentiment analysis
            sentiment = get_sentiment(article['text'])
            # Skip the article if its sentiment is neutral
            if sentiment == "neutral":
                print(f"Skipping article due to neutral sentiment: {article['url']}")
                continue

            # Extract stock prices for evaluation
            open_price = stock_data.loc[day_after, 'Open']
            close_price = stock_data.loc[day_after, 'Close']
            day_after_percentage = ((close_price - open_price) / open_price) * 100
            day_after_win = close_price > open_price if sentiment == "positive" else close_price < open_price

            # Find the trading day X days after the 'day_after'
            x_days_after = next((d for i, d in enumerate(stock_trading_dates[stock_trading_dates > day_after]) if i == lookahead_days - 1), None)
            x_day_percentage = None
            x_day_win = None
            if x_days_after:
                future_close_price = stock_data.loc[x_days_after, 'Close']
                x_day_percentage = ((future_close_price - open_price) / open_price) * 100
                x_day_win = future_close_price > open_price if sentiment == "positive" else future_close_price < open_price

            # Append evaluation results including the publish_date
            results.append({
                'url': article['url'],
                'publish_date': publish_date,  # Include publish date here
                'sentiment': sentiment,
                'day_after_win': day_after_win,
                'day_after_percentage': day_after_percentage,
                'x_day_win': x_day_win,
                'x_day_percentage': x_day_percentage,
            })

        except Exception as e:
            logging.error(f"Error evaluating article {article['url']}: {e}")
            print(f"Error evaluating article {article['url']}: {e}")

    return results

def calculate_metrics(results):
    """
    Calculate and print interesting metrics based on evaluation results.

    Parameters:
        results (list): Evaluation results containing URLs, sentiment, and win outcomes.

    Returns:
        dict: A dictionary of calculated metrics.
    """
    metrics = {
        'total_articles': len(results),
        'sentiment_distribution': {'positive': 0, 'negative': 0},
        'day_after_success_rate': 0,
        'x_day_success_rate': 0,
        'average_day_percentage': 0,
        'average_x_day_percentage': 0,
        'cumulative_day_percentage': 0,
        'cumulative_x_day_percentage': 0,
        'first_trade_date': None,
        'last_trade_date': None,
        'total_days_between_trades': 0,
        'sentiment_based_success': {'positive': {'day_after': 0, 'x_day': 0},
                                    'negative': {'day_after': 0, 'x_day': 0}},
        'sentiment_counts': {'positive': 0, 'negative': 0},
    }

    day_after_success = 0
    x_day_success = 0
    day_percentages = []
    x_day_percentages = []
    trade_dates = []

    for result in results:
        sentiment = result['aggregated_sentiment']

        # Collect trade dates
        trade_dates.append(result['publish_date'])

        # Collect day-after and X-day percentages
        day_percent = result['day_after_percentage']
        day_percentages.append(day_percent)

        x_percent = result['x_day_percentage']
        if x_percent is not None:
            x_day_percentages.append(x_percent)

        # Ensure day_after_win and x_day_win are scalars
        day_after_win = (
            result['day_after_win'].iloc[0]
            if isinstance(result['day_after_win'], pd.Series)
            else result['day_after_win']
        )
        x_day_win = (
            result['x_day_win'].iloc[0]
            if isinstance(result['x_day_win'], pd.Series)
            else result['x_day_win']
        )

        # Update sentiment distribution
        metrics['sentiment_distribution'][sentiment] += 1

        # Count successes
        if day_after_win:
            day_after_success += 1
            metrics['sentiment_based_success'][sentiment]['day_after'] += 1
        if x_day_win:
            x_day_success += 1
            metrics['sentiment_based_success'][sentiment]['x_day'] += 1

        # Count sentiments
        metrics['sentiment_counts'][sentiment] += 1

    # Convert trade dates to actual dates and find first/last trade dates
    if trade_dates:
        trade_dates = sorted([datetime.datetime.strptime(date, "%Y-%m-%d") for date in trade_dates])
        metrics['first_trade_date'] = trade_dates[0].strftime("%Y-%m-%d")
        metrics['last_trade_date'] = trade_dates[-1].strftime("%Y-%m-%d")
        metrics['total_days_between_trades'] = (trade_dates[-1] - trade_dates[0]).days

    # Calculate success rates
    metrics['day_after_success_rate'] = day_after_success / metrics['total_articles'] if metrics['total_articles'] > 0 else 0
    metrics['x_day_success_rate'] = x_day_success / metrics['total_articles'] if metrics['total_articles'] > 0 else 0

    # Calculate average percentages
    metrics['average_day_percentage'] = np.mean(day_percentages) if day_percentages else 0
    metrics['average_x_day_percentage'] = np.mean(x_day_percentages) if x_day_percentages else 0

    # Calculate cumulative percentages
    metrics['cumulative_day_percentage'] = np.sum(day_percentages) if day_percentages else 0
    metrics['cumulative_x_day_percentage'] = np.sum(x_day_percentages) if x_day_percentages else 0

    return metrics

def group_daily_news(articles):
    """
    Groups the articles by their publish_date and computes the mean polarity for each day.
    Then, it determines the aggregated sentiment for that day:
      - 'positive' if mean polarity > 0.05
      - 'negative' if mean polarity < -0.05
      - 'neutral' otherwise

    Parameters:
        articles (list): List of article dicts that have at least the keys 'publish_date' and 'text'.

    Returns:
        list: A list of daily aggregated news entries (dicts) with keys:
              'publish_date', 'mean_polarity', and 'aggregated_sentiment'.
              The list is sorted in chronological order.
    """
    daily = {}
    for art in articles:
        date = art.get('publish_date')
        if not date:
            continue
        try:
            # Calculate the polarity of the article text
            polarity = TextBlob(art['text']).sentiment.polarity
        except Exception as e:
            logging.error(f"Error calculating polarity for article {art.get('url', '')}: {e}")
            continue
        daily.setdefault(date, []).append(polarity)

    daily_list = []
    for date, polarities in daily.items():
        mean_pol = np.mean(polarities)
        if mean_pol > 0.05:
            sentiment = 'positive'
        elif mean_pol < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        daily_list.append({
            'publish_date': date,
            'mean_polarity': mean_pol,
            'aggregated_sentiment': sentiment
        })

    # Sort in chronological order (YYYY-MM-DD sorts correctly as a string)
    daily_list.sort(key=lambda x: x['publish_date'])
    return daily_list

def evaluate_trades_daily(daily_news, stock_data, lookahead_days=5):
    """
    For each day (from the aggregated daily news), evaluates the trading outcome.
    Only days with aggregated sentiment that is 'positive' or 'negative' are kept.
    The trading outcome is computed using the next trading day and X-days after.
    The results are sorted in chronological order.

    Parameters:
        daily_news (list): List of daily aggregated news entries (from group_daily_news).
        stock_data (pandas.DataFrame): Historical stock data.
        lookahead_days (int): Number of days to look ahead for evaluation.

    Returns:
        list: A list of daily trading evaluation results with keys:
              'publish_date', 'aggregated_sentiment', 'mean_polarity',
              'day_after_win', 'day_after_percentage', 'x_day_win', and 'x_day_percentage'.
    """
    results = []
    stock_trading_dates = stock_data.index

    for daily in daily_news:
        # Skip days with neutral aggregated sentiment
        if daily['aggregated_sentiment'] == 'neutral':
            continue

        publish_date = daily['publish_date']
        try:
            news_date = datetime.datetime.strptime(publish_date, "%Y-%m-%d").date()
        except Exception as e:
            print(f"Skipping daily news with invalid publish date: {publish_date}")
            continue

        # Find the next trading day after the news date
        day_after = next((d for d in stock_trading_dates if d.date() > news_date), None)
        if not day_after:
            print(f"No trading data available after {news_date}. Skipping daily news for date: {publish_date}")
            continue

        sentiment = daily['aggregated_sentiment']

        # Get open and close prices on the day after; convert to scalar if needed.
        open_price = stock_data.loc[day_after, 'Open']
        if isinstance(open_price, pd.Series):
            open_price = open_price.iloc[0]
        close_price = stock_data.loc[day_after, 'Close']
        if isinstance(close_price, pd.Series):
            close_price = close_price.iloc[0]

        day_after_percentage = ((close_price - open_price) / open_price) * 100
        day_after_win = (close_price > open_price) if sentiment == 'positive' else (close_price < open_price)

        # Find the trading day X days after the 'day_after'
        x_days_after = next((d for i, d in enumerate(stock_trading_dates[stock_trading_dates > day_after])
                             if i == lookahead_days - 1), None)
        x_day_percentage = None
        x_day_win = None
        if x_days_after:
            future_close_price = stock_data.loc[x_days_after, 'Close']
            if isinstance(future_close_price, pd.Series):
                future_close_price = future_close_price.iloc[0]
            x_day_percentage = ((future_close_price - open_price) / open_price) * 100
            x_day_win = (future_close_price > open_price) if sentiment == 'positive' else (
                        future_close_price < open_price)

        results.append({
            'publish_date': publish_date,
            'aggregated_sentiment': sentiment,
            'mean_polarity': daily['mean_polarity'],
            'day_after_win': day_after_win,
            'day_after_percentage': day_after_percentage,
            'x_day_win': x_day_win,
            'x_day_percentage': x_day_percentage
        })

    # Sort results in chronological order
    results.sort(key=lambda x: x['publish_date'])
    return results

# Main Execution
print(f"Keywords: {KEYWORDS}")
articles_file = "matched_articles.json"

if os.path.exists(articles_file):
    articles = load_articles_from_file(articles_file)
else:
    articles = fetch_news_articles(warc_folder, KEYWORDS)
    print(f"Found {len(articles)} articles.")
    save_articles_to_file(articles, articles_file)

# Continue with your grouping and evaluation as before:
daily_news = group_daily_news(articles)
print(f"Aggregated into {len(daily_news)} daily news entries.")

dt_start_pad = (datetime.datetime.strptime(START_DATE_NEWS, "%Y-%m-%d") - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
dt_end_pad = (datetime.datetime.strptime(END_DATE_NEWS, "%Y-%m-%d") + datetime.timedelta(days=10 + LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")
stock_data = get_stock_data(TICKER, dt_start_pad, dt_end_pad)

daily_results = evaluate_trades_daily(daily_news, stock_data, LOOKAHEAD_DAYS)
print("\n=== DAILY TRADE EVALUATION RESULTS ===")
for r in daily_results:
    print(f"\nPublish Date: {r['publish_date']}")
    print(f"Aggregated Sentiment: {r['aggregated_sentiment']} (Mean Polarity: {r['mean_polarity']:.3f})")
    print(f"Day-after Win: {r['day_after_win']}")
    print(f"Day-after %: {r['day_after_percentage']:.2f}%")
    if r['x_day_percentage'] is not None:
        print(f"{LOOKAHEAD_DAYS}-day %: {r['x_day_percentage']:.2f}%")
    else:
        print(f"{LOOKAHEAD_DAYS}-day %: N/A")



    # Print the snippet from the article
    if 'snippet' in r and r['snippet']:
        print(f"Snippet: {r['snippet']}\n")

# Calculate and print metrics
metrics = calculate_metrics(daily_results)

print("\n=== METRICS ===")
print(f"Total Articles (Total Trades): {metrics['total_articles']}")
print(f"Sentiment Distribution: {metrics['sentiment_distribution']}")
print(f"Day-after Success Rate: {metrics['day_after_success_rate']:.2%}")
print(f"{LOOKAHEAD_DAYS}-day Success Rate: {metrics['x_day_success_rate']:.2%}")
print(f"Average Day-after % Change: {metrics['average_day_percentage']:.2f}%")
print(f"Average {LOOKAHEAD_DAYS}-day % Change: {metrics['average_x_day_percentage']:.2f}%")
print(f"Cumulative Day-after % Change: {metrics['cumulative_day_percentage']:.2f}%")
print(f"Cumulative {LOOKAHEAD_DAYS}-day % Change: {metrics['cumulative_x_day_percentage']:.2f}%")
print(f"First Trade Date: {metrics['first_trade_date']}")
print(f"Last Trade Date: {metrics['last_trade_date']}")
print(f"Total Days Between First and Last Trade: {metrics['total_days_between_trades']} days")
print("Sentiment-based Success Rates:")
for sentiment, data in metrics['sentiment_based_success'].items():
    print(f"  {sentiment.capitalize()}: Day-after: {data['day_after']}, X-day: {data['x_day']}")

