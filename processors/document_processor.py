from langchain.schema.document import Document

class DocumentProcessor:
    """
    將文本 chunk 轉換為 langchain Document 物件
    """
    def __init__(self, chunks):
        self.chunks = chunks

    def to_documents(self):
        """
        將每個文本 chunk 轉換成 Document 物件
        """
        return [Document(page_content=chunk) for chunk in self.chunks]
