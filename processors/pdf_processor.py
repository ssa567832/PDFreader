from hashlib import sha256
import pymupdf4llm
import pymupdf.pro
from langchain.text_splitter import MarkdownHeaderTextSplitter

class PDFProcessor:
    """
    處理 PDF 轉換與 Markdown 文本分割
    """
    def __init__(self, pdf_path, headers_to_split_on, unlock_pro=True):
        """
        :param pdf_path: PDF 文件路徑
        :param headers_to_split_on: Markdown 標題的切分規則
        :param unlock_pro: 是否啟用 pymupdf 的 Pro 功能
        """
        self.pdf_path = pdf_path
        self.headers_to_split_on = headers_to_split_on
        self.unlock_pro = unlock_pro

    def extract_markdown(self):
        """
        解鎖 Pro 功能後，將 PDF 轉換為 Markdown 格式
        """
        if self.unlock_pro:
            pymupdf.pro.unlock()
        md_text = pymupdf4llm.to_markdown(self.pdf_path)
        return md_text

    def split_markdown(self, md_text):
        """
        根據指定的 Markdown 標題進行文本切分
        """
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        chunks = splitter.split_text(md_text)
        # 過濾空的 chunk 並轉為字串
        return [str(chunk) for chunk in chunks if chunk]

    def deduplicate_chunks(self, chunks):
        """
        使用 SHA256 去除重複的 chunk
        """
        unique_chunks = []
        seen_hashes = set()
        for chunk in chunks:
            chunk_hash = sha256(chunk.encode('utf-8')).hexdigest()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        return unique_chunks
