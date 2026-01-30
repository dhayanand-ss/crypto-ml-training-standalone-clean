
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data():
    print("Generating sample data...")
    
    # Ensure directories exist
    os.makedirs("data/prices", exist_ok=True)
    os.makedirs("data/articles", exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Generate Price Data (BTCUSDT_sample.csv)
    # ---------------------------------------------------------
    # Generate 1000 data points (approx 16 hours of minute data)
    n_samples = 1000
    end_date = datetime.now()
    start_date = end_date - timedelta(minutes=n_samples)
    
    dates = [start_date + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate random price walk
    base_price = 100000.0
    price_changes = np.random.normal(0, 50, n_samples)
    prices = base_price + np.cumsum(price_changes)
    
    # Create DataFrame with ALL required columns
    df_prices = pd.DataFrame({
        'open_time': [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates],
        'open': prices + np.random.normal(0, 10, n_samples),
        'high': prices + np.abs(np.random.normal(0, 20, n_samples)),
        'low': prices - np.abs(np.random.normal(0, 20, n_samples)),
        'close': prices,
        'volume': np.abs(np.random.normal(100, 50, n_samples)),
        'quote_asset_volume': np.abs(np.random.normal(1000000, 500000, n_samples)), # Filled!
        'trades': np.random.randint(10, 100, n_samples), # Filled!
        'taker_base': np.abs(np.random.normal(50, 25, n_samples)), # Filled!
        'taker_quote': np.abs(np.random.normal(500000, 250000, n_samples)), # Filled!
        'ignore': 0,
        'date': [d.strftime('%Y-%m-%d') for d in dates]
    })
    
    # Save prices
    price_path = "data/prices/BTCUSDT.csv"
    df_prices.to_csv(price_path, index=False)
    print(f"Created {price_path} with {len(df_prices)} rows")
    
    # Save test prices (subset)
    test_path = "data/prices/BTCUSDT_test.csv"
    df_prices.iloc[-200:].to_csv(test_path, index=False)
    print(f"Created {test_path} with 200 rows")
    
    # ---------------------------------------------------------
    # 2. Generate Articles Data (articles_sample.csv)
    # ---------------------------------------------------------
    # Generate articles for the overlapping dates
    unique_dates = df_prices['date'].unique()
    
    article_data = []
    sentiments = ['positive', 'negative', 'neutral']
    
    for date in unique_dates:
        # Generate 5-10 articles per day
        n_articles = np.random.randint(5, 11)
        for i in range(n_articles):
            article_data.append({
                'date': date,
                'title': f"Sample crypto news headline {i} for {date}",
                'text': f"Sample content for article {i} on {date}. This is just a dummy text for checking pipeline.",
                'sentiment': np.random.choice(sentiments),
                'source': 'sample_source',
                'link': f'http://sample.url/{i}'
            })
            
    df_articles = pd.DataFrame(article_data)
    
    # Save articles
    article_path = "data/articles/articles.csv" # Fixed name
    df_articles.to_csv(article_path, index=False)
    print(f"Created {article_path} with {len(df_articles)} rows")

if __name__ == "__main__":
    generate_sample_data()
