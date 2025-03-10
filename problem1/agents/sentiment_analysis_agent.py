from langchain.agents import Agent

class SentimentAnalysisAgent(Agent):
    def __init__(self, embedding_tool, sentiment_tool):
        self.embedding_tool = embedding_tool
        self.sentiment_tool = sentiment_tool

    def _call(self, inputs):
        texts = inputs["texts"]
        embeddings = self.embedding_tool({"texts": texts})["embeddings"]
        sentiment_scores = self.sentiment_tool({"texts": texts})["sentiment_scores"]
        return {"texts": texts, "embeddings": embeddings, "sentiment_scores": sentiment_scores}
