import openai, os
from tools.embedding_tool import EmbeddingTool
from tools.sentiment_analysis_tool import SentimentAnalysisTool
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from database.db_operations import connect_to_db, create_table, insert_data
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Instantiate tools
embedding_tool = EmbeddingTool()
sentiment_tool = SentimentAnalysisTool()

# Instantiate agent
agent = SentimentAnalysisAgent(embedding_tool, sentiment_tool)

# Example texts
tweets = [
    "Achieving work-life balance is crucial for mental health.",
    "Work-life balance is a myth. You have to make sacrifices.",
    "I finally found a job that respects my work-life balance!",
    "Struggling to maintain work-life balance during the pandemic.",
    "Work-life balance is not just about time management, it's about energy management."
]

# Run the agent
results = agent({"texts": tweets})

# Extract results
texts = results["texts"]
embeddings = results["embeddings"]
sentiment_scores = results["sentiment_scores"]

# Connect to PostgreSQL
conn = connect_to_db()

# Create a table to store vectors and sentiment scores
create_table(conn)

# Insert vectors and sentiment scores into the table
insert_data(conn, texts, embeddings, sentiment_scores)

print("Data inserted successfully.")
