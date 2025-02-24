import os
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    """
    負責 FAISS 向量數據庫的加載、更新與儲存
    """
    def __init__(self, db_path, embedding_function):
        """
        :param db_path: FAISS 數據庫保存路徑
        :param embedding_function: 嵌入函數（例如 HuggingFaceEmbeddings）
        """
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.vectorstore = None
        self.load_or_create_vectorstore()

    def load_or_create_vectorstore(self):
        """
        根據是否存在索引文件，選擇加載或建立 FAISS 數據庫
        """
        index_file = os.path.join(self.db_path, "index.faiss")
        if os.path.exists(self.db_path) and os.path.exists(index_file):
            try:
                print("Loading existing FAISS database...")
                self.vectorstore = FAISS.load_local(
                    self.db_path, 
                    self.embedding_function,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Error loading FAISS database: {e}. Creating a new FAISS database...")
                self._create_new_vectorstore()
        else:
            print("Creating a new FAISS database...")
            self._create_new_vectorstore()

    def _create_new_vectorstore(self):
        """
        建立一個新的空 FAISS 數據庫
        """
        # 取得一個 sample embedding 以決定向量維度
        sample_embedding = self.embedding_function.embed_query("sample")
        dimension = len(sample_embedding)
        # 建立 FAISS 平面 L2 索引
        index = faiss.IndexFlatL2(dimension)
        # 使用 InMemoryDocstore 作為底層 docstore（它支援添加項目）
        from langchain.docstore.in_memory import InMemoryDocstore
        docstore = InMemoryDocstore({})
        # 建立空的 index_to_docstore_id 映射
        self.vectorstore = FAISS(
            self.embedding_function, 
            index, 
            docstore, 
            {}
        )

    def deduplicate_documents(self, documents):
        """
        與現有數據庫中的嵌入比對，去除重複的文檔
        """
        unique_documents = []
        if self.vectorstore.index.ntotal > 0:
            existing_embeddings = np.array(
                self.vectorstore.index.reconstruct_n(0, self.vectorstore.index.ntotal)
            )
            for doc in documents:
                doc_embedding = self.embedding_function.embed_query(doc.page_content)
                if not np.any(np.all(np.isclose(existing_embeddings, doc_embedding, atol=1e-5), axis=1)):
                    unique_documents.append(doc)
        else:
            unique_documents = documents
        return unique_documents

    def add_documents(self, documents):
        """
        將新的文檔添加到向量數據庫中
        """
        if documents:
            self.vectorstore.add_documents(documents)
            print("New documents added to the FAISS database.")
        else:
            print("No unique documents to add.")

    def save_vectorstore(self):
        """
        保存更新後的 FAISS 數據庫
        """
        self.vectorstore.save_local(self.db_path)
        print("Updated FAISS database with unique documents has been saved.")
