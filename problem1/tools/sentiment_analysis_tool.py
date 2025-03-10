from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class SentimentAnalysisTool(Tool):
    def __init__(self, model_name='text-davinci-003'):
        self.model = OpenAI(engine=model_name)
        self.prompt_template = PromptTemplate(
            input_variables=["text"],
            template="Analyze the sentiment of the following text and provide a score between -1 (very negative) and 1 (very positive):\n\n{text}\n\nSentiment score:"
        )

    def _call(self, inputs):
        texts = inputs["texts"]
        sentiment_scores = []
        for text in texts:
            response = self.model(self.prompt_template.format(text=text))
            sentiment_score = float(response.strip())
            sentiment_scores.append(sentiment_score)
        return {"sentiment_scores": sentiment_scores}
