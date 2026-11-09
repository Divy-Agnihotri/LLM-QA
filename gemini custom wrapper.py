"""
code to use gemini as llm as judge, by default deep eval uses OPENAI, so a custom wrap is required for using local deployed model, this is for gemini
"""
#import os
#os.environ["GOOGLE_API_KEY"] = "add api key here"

from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.models.base_model import DeepEvalBaseLLM
class LCGemini(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    def get_model_name(self):
        return "Custom Azure OpenAI Model"

custom_model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.7
)
lcGemini = LCGemini(model=custom_model)
print(lcGemini.generate("Write me a joke"))
