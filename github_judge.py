#Create GitHub Model Judge
import os

# 🔑 MUST be before ChatOpenAI is imported or used
os.environ["OPENAI_API_KEY"] = os.environ.get(
    "GITHUB_TOKEN",
    "ghp_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
)

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI


class GitHubModelJudge(DeepEvalBaseLLM):
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    def load_model(self):
        return ChatOpenAI(
            model=self.model_name,
            base_url="https://models.inference.ai.azure.com",
            temperature=0
        )

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return f"GitHub Models: {self.model_name}"
