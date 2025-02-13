import os
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from textblob import TextBlob
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup
import nltk
import logging
import time

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
TICKER = "AAPL" #"MSFT"
START_DATE_NEWS = "2010-10-01"
END_DATE_NEWS = "2025-01-01"
LOOKAHEAD_DAYS = 5
warc_file_path = r"C:\Users\yoelarie\Downloads\wget-1.21.4-win64\downloaded_warc_files\CC-MAIN-20241201162023-20241201192023-00001.warc\file.warc"

COMPANY_NAME = get_company_name(TICKER)
PRIMARY_NAME = get_primary_name(COMPANY_NAME)
KEYWORDS = [TICKER, COMPANY_NAME, PRIMARY_NAME]  # Include TICKER, full name, and primary name

# --------------------------------------------
# STEP 1: FETCH NEWS ARTICLES FROM WARC
# --------------------------------------------
def extract_main_text(html_content):
    """
    Extracts the main text content from HTML using BeautifulSoup.

    Parameters:
        html_content (str): Raw HTML content.

    Returns:
        str: Extracted main text or None if no content found.
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        article_text = " ".join(p.get_text() for p in soup.find_all("p"))
        return article_text if article_text else None
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return None


def extract_snippet(text, keyword, window=50):
    """
    Extracts a snippet of text around the first occurrence of 'keyword'.

    Parameters:
        text (str): The text to search.
        keyword (str): The keyword to locate.
        window (int): Number of characters to include before and after the keyword.

    Returns:
        str: The snippet around the keyword or an empty string if not found.
    """
    lower_text = text.lower()
    lower_keyword = keyword.lower()
    index = lower_text.find(lower_keyword)
    if index != -1:
        start = max(0, index - window)
        end = index + len(keyword) + window
        return text[start:end]
    return ""


def fetch_news_articles(warc_path, keywords, max_records=500):
    articles = []
    record_count = 0
    total_records = sum(1 for _ in ArchiveIterator(open(warc_path, 'rb')))

    try:
        with open(warc_path, 'rb') as warc_file:
            for i, record in enumerate(ArchiveIterator(warc_file)):
                if record_count >= max_records:
                    break
                if record.rec_type != 'response':
                    continue

                try:
                    url = record.rec_headers.get_header("WARC-Target-URI")
                    content = record.content_stream().read()
                    html_text = content.decode("utf-8", errors="ignore")

                    # Extract main text
                    main_text = extract_main_text(html_text)
                    publish_date = None

                    # Attempt to extract a publish date from metadata
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

                    # Filter and store matching articles
                    if main_text and any(keyword.lower() in main_text.lower() for keyword in keywords):
                        # Extract a snippet using the first matching keyword
                        snippet = ""
                        for kw in keywords:
                            snippet = extract_snippet(main_text, kw, window=50)
                            if snippet:
                                break
                        articles.append({
                            'url': url,
                            'text': main_text[:1000],  # Limit to 1000 characters for preview
                            'snippet': snippet,
                            'publish_date': publish_date,
                        })
                        record_count += 1
                        print(f"Matched article {record_count}: {url}")
                        print(f"Publish Date: {publish_date}")
                        print(f"Snippet: {snippet}\n")
                except Exception as e:
                    logging.error(f"Error processing WARC record: {e}")

                # Print progress every 100 records processed
                if i % 100 == 0:
                    print(f"Processed {i} records out of {total_records} so far...")

    except FileNotFoundError:
        logging.error(f"WARC file not found: {warc_path}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    print(f"Finished processing WARC file. Found {len(articles)} matching articles.")
    return articles



# --------------------------------------------
# STEP 2: PERFORM SENTIMENT ANALYSIS
# --------------------------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
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
                # If no publish_date is available, skip the article
                print(f"Skipping article with missing or invalid publish date: {article['url']}")
                continue

            # Find the next trading day (day after the news)
            day_after = next((d for d in stock_trading_dates if d.date() > news_date), None)
            if not day_after:
                print(f"No trading data available after {news_date}. Skipping article: {article['url']}")
                continue

            # Perform sentiment analysis
            sentiment = get_sentiment(article['text'])

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

            # Append evaluation results
            results.append({
                'url': article['url'],
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
        'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
        'day_after_success_rate': 0,
        'x_day_success_rate': 0,
        'sentiment_based_success': {'positive': {'day_after': 0, 'x_day': 0},
                                    'neutral': {'day_after': 0, 'x_day': 0},
                                    'negative': {'day_after': 0, 'x_day': 0}},
        'sentiment_counts': {'positive': 0, 'neutral': 0, 'negative': 0},
    }

    day_after_success = 0
    x_day_success = 0

    for result in results:
        sentiment = result['sentiment']

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

    # Calculate success rates
    metrics['day_after_success_rate'] = day_after_success / metrics['total_articles'] if metrics['total_articles'] > 0 else 0
    metrics['x_day_success_rate'] = x_day_success / metrics['total_articles'] if metrics['total_articles'] > 0 else 0

    return metrics



# Main Execution
print(f"Keywords: {KEYWORDS}")
articles = fetch_news_articles(warc_file_path, KEYWORDS)
print(f"Found {len(articles)} articles.")

dt_start_pad = (datetime.datetime.strptime(START_DATE_NEWS, "%Y-%m-%d") - datetime.timedelta(days=10)).strftime("%Y-%m-%d")
dt_end_pad = (datetime.datetime.strptime(END_DATE_NEWS, "%Y-%m-%d") + datetime.timedelta(days=10 + LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")
stock_data = get_stock_data(TICKER, dt_start_pad, dt_end_pad)

results = evaluate_trades(articles, stock_data, LOOKAHEAD_DAYS)
print("\n=== TRADE EVALUATION RESULTS ===")
for r in results:
    # Handle scalar values or multi-ticker (Series) correctly
    day_after_percentage = (
        r['day_after_percentage'].iloc[0]  # Use first value if it's a Series
        if isinstance(r['day_after_percentage'], pd.Series)
        else r['day_after_percentage']
    )
    x_day_percentage = (
        r['x_day_percentage'].iloc[0]
        if isinstance(r['x_day_percentage'], pd.Series) and r['x_day_percentage'] is not None
        else r['x_day_percentage']
    )

    print(f"\nURL: {r['url']}\n"
          f"Sentiment: {r['sentiment']}\n"
          f"Day-after Win: {r['day_after_win']}\n"
          f"Day-after %: {day_after_percentage:.2f}%\n"
          f"{LOOKAHEAD_DAYS}-day Win: {r['x_day_win']}\n"
          f"{LOOKAHEAD_DAYS}-day %: {x_day_percentage:.2f}%" if x_day_percentage is not None else "X-day %: N/A")

    # Print the snippet from the article
    if 'snippet' in r and r['snippet']:
        print(f"Snippet: {r['snippet']}\n")

# Calculate and print metrics
metrics = calculate_metrics(results)

print("\n=== METRICS ===")
print(f"Total Articles: {metrics['total_articles']}")
print(f"Sentiment Distribution: {metrics['sentiment_distribution']}")
print(f"Day-after Success Rate: {metrics['day_after_success_rate']:.2%}")
print(f"{LOOKAHEAD_DAYS}-day Success Rate: {metrics['x_day_success_rate']:.2%}")
print("Sentiment-based Success Rates:")
for sentiment, data in metrics['sentiment_based_success'].items():
    print(f"  {sentiment.capitalize()}: Day-after: {data['day_after']}, X-day: {data['x_day']}")
