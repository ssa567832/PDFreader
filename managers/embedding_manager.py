from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    """
    負責初始化嵌入模型（例如 HuggingFaceEmbeddings）
    """
    def __init__(self, model_name: str, model_kwargs: dict = None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.embedding_function = None

    def initialize_embeddings(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs
        )
        return self.embedding_function
