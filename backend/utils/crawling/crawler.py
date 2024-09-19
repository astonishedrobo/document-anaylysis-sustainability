import datetime
import json
from gnews import GNews
import os
from newspaper import Article, Config
import threading
import requests


def date_range(start_date: datetime.date, end_date: datetime.date, delta: int):
    delta_ = datetime.timedelta(days=delta)
    while start_date <= end_date:
        yield (start_date.timetuple()[:3], (start_date + datetime.timedelta(days=6)).timetuple()[:3])
        start_date += delta_

def get_final_url(url):
    response = requests.get(url, allow_redirects=True)
    return response.url

def get_full_article(news: dict):
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    config = Config()
    config.browser_user_agent = user_agent
    article = Article(get_final_url(news['url']), config=config)
    article.download()
    article.parse()

    return {'text': article.text, 'title': article.title}

def fetch_news(start_date: tuple, end_date: tuple, query: str, fetch_article: bool = False):
    g_news = GNews(country='IN', start_date=start_date, end_date=end_date)
    news = g_news.get_news(query)
    
    if fetch_article:
        for news_ in news:
            news_['article'] = get_full_article(news_)
    return news

def save_news(news: list, file_lock: threading.Lock, filename='news.json'):
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    if not news:
        return

    with file_lock:
        try:
            with open(f'./data/{filename}', 'r') as f:
                existing_news = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_news = []

        existing_news.extend(news)

        # Save all news to the JSON file
        with open(f'./data/{filename}', 'w') as f:
            json.dump(existing_news, f, ensure_ascii=False, indent=4)