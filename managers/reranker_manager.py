from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

class RerankerManager:
    """
    負責初始化第二階段的 reranker
    """
    def __init__(self, top_n: int = 7, model: str = "BAAI/bge-reranker-large", use_fp16: bool = True):
        self.top_n = top_n
        self.model = model
        self.use_fp16 = use_fp16
        self.reranker = None

    def initialize_reranker(self):
        self.reranker = FlagEmbeddingReranker(
            top_n=self.top_n,
            model=self.model,
            use_fp16=self.use_fp16
        )
        return self.reranker
