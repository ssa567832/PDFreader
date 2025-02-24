from langchain.embeddings import HuggingFaceEmbeddings
from processors.pdf_processor import PDFProcessor
from processors.document_processor import DocumentProcessor
from managers.vectorstore_manager import VectorStoreManager

class PreprocessingPipeline:
    """
    前處理流程管道：
    PDF 轉 Markdown、文本切分、去重、轉 Document 以及更新 FAISS 向量資料庫
    """
    def __init__(self, pdf_path, db_path, embedding_function=None, model_name="intfloat/multilingual-e5-large", device="cpu", unlock_pro=True):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.model_name = model_name
        self.device = device
        self.unlock_pro = unlock_pro

        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]

        # 如果外部已經提供 embedding_function，則直接使用；否則初始化新的嵌入模型
        if embedding_function is not None:
            self.embedding_function = embedding_function
        else:
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device}
            )
            
        self.pdf_processor = PDFProcessor(
            pdf_path=self.pdf_path,
            headers_to_split_on=self.headers_to_split_on,
            unlock_pro=self.unlock_pro
        )
        self.vector_store_manager = VectorStoreManager(self.db_path, self.embedding_function)

    def run(self):
        # 1. PDF 轉 Markdown
        md_text = self.pdf_processor.extract_markdown()

        # 2. Markdown 切分為 chunks
        chunks = self.pdf_processor.split_markdown(md_text)

        # 3. 去除重複的 chunks
        unique_chunks = self.pdf_processor.deduplicate_chunks(chunks)
        print(f"Number of unique chunks: {len(unique_chunks)}")
        for chunk in unique_chunks:
            print(chunk)

        # 4. 轉換為 Document 物件
        doc_processor = DocumentProcessor(unique_chunks)
        documents = doc_processor.to_documents()

        # 5. 將文件加入 FAISS 向量庫並保存
        self.vector_store_manager.add_documents(documents)
        self.vector_store_manager.save_vectorstore()
