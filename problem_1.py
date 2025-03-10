import tweepy
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import psycopg2
from pgvector.psycopg2 import register_vector
import json
# Twitter API credentials
consumer_key = '5O7GaZV7UYKI6p4KgY1TGf4nF'
consumer_secret = 'edEnoDVbX1iJAS5L3s6zkS31jIKqFfWmrQ2PTi3HKByHYyTu5i'
access_token = '1897967854924005376-HtYtcVHDtCW2wOUk0QZVLhKaYfuy5a'
access_token_secret = 'Nu4LT7M6JdsYR50r9IWKzHbkft0ElB4cc1yu7AUWBECAp'

# Twitter API v2 credentials
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAQfzwEAAAAA0BdjfYqFLV4TVPHytQzBzK%2BhbEA%3DZ2FuUvNTJBb4xc1rhOLa84zvTBYqerQBXTEZxvQWTKdQHKXmm2'

# Authenticate to Twitter API v2
# client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
#
# # Fetch tweets using Twitter API v2
# query = "work-life balance"
# tweets = client.search_recent_tweets(query=query, max_results=100)
#
# # Extract tweet texts
# tweet_texts = [tweet.text for tweet in tweets.data]
# print(tweet_texts)

#using basic tweets as Rate limiting error from tweepy
tweet_texts = [
    "Achieving work-life balance is crucial for mental health.",
    "Work-life balance is a myth. You have to make sacrifices.",
    "I finally found a job that respects my work-life balance!",
    "Struggling to maintain work-life balance during the pandemic.",
    "Work-life balance is not just about time management, it's about energy management.",
    "Companies need to prioritize work-life balance for their employees.",
    "Work-life balance is different for everyone. Find what works for you.",
    "Remote work has improved my work-life balance significantly.",
    "Work-life balance is essential for productivity and happiness.",
    "Finding work-life balance is an ongoing journey, not a destination."
]

# # Extract article texts
# articles = soup.find_all('div')
# article_texts = [article.get_text() for article in articles]
# print(article_texts)
# Combine and preprocess data

all_texts = tweet_texts
print(all_texts)
def preprocess_text(text):
    return text.lower().replace('\n', ' ').replace('\r', '')

processed_texts = [preprocess_text(text) for text in all_texts]

# Convert texts to vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(processed_texts).toarray()
print(len(vectors))
# Connect to PostgreSQL
conn = psycopg2.connect(
    database="mydatabase",
    user="docker_user",
    password="docker_user",
    host="localhost",
    port="5433"
)

# Register pgvector extension
register_vector(conn)
#
# # Create a table to store vectors
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS work_life_balance_vectors (
            id SERIAL PRIMARY KEY,
            text TEXT,
            vector VECTOR(56)
        )
    """)
    conn.commit()

# Insert vectors into the table
with conn.cursor() as cur:
    for text, vector in zip(processed_texts, vectors):
        cur.execute("""
            INSERT INTO work_life_balance_vectors (text, vector)
            VALUES (%s, %s)
        """, (text, vector.tolist()))
    conn.commit()

# Query the vector database
query_text = "How to achieve work-life balance?"
query_vector = vectorizer.transform([preprocess_text(query_text)]).toarray()[0]
# print(query_vector)

with conn.cursor() as cur:
    cur.execute("""
        SELECT text
        FROM work_life_balance_vectors
        ORDER BY vector <=> %s::vector
        LIMIT 5
    """, (query_vector.tolist(),))
    top_texts = cur.fetchall()

print(top_texts)
# Print the top results
for text in top_texts:
    print(text[0])