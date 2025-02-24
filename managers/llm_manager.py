from langchain_community.llms import Ollama

class LLMManager:
    """
    負責初始化 LLM（例如 Ollama）
    """
    def __init__(self, api_base: str, model_name: str):
        self.api_base = api_base
        self.model_name = model_name
        self.llm = None

    def initialize_llm(self):
        self.llm = Ollama(base_url=self.api_base, model=self.model_name)
        return self.llm
