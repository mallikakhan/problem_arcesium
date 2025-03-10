import os
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def connect_to_db():
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    register_vector(conn)
    return conn

def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS work_life_balance_vectors (
                id SERIAL PRIMARY KEY,
                text TEXT,
                vector VECTOR(384),
                sentiment_score FLOAT
            )
        """)
        conn.commit()

def insert_data(conn, texts, embeddings, sentiment_scores):
    with conn.cursor() as cur:
        for text, vector, score in zip(texts, embeddings, sentiment_scores):
            cur.execute("""
                INSERT INTO work_life_balance_vectors (text, vector, sentiment_score)
                VALUES (%s, %s, %s)
            """, (text, vector.tolist(), score))
        conn.commit()
