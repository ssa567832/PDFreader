from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

class QAPipeline:
    """
    問答流程管道：
    1. 向量檢索初步獲取文件
    2. 利用 reranker 重排序獲得上下文
    3. 用 LLMChain 根據上下文生成答案
    """
    def __init__(self, llm, vectorstore, reranker):
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.answer_chain = self.create_answer_chain()
    
    def create_answer_chain(self):
        answer_template = """
        Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        return LLMChain(llm=self.llm, prompt=PromptTemplate.from_template(answer_template))
    
    def retrieve_context(self, question: str, top_k: int = 30, top_n_rerank: int = 7):
        try:
            # 第一階段：向量檢索
            initial_docs = self.vectorstore.similarity_search(question, k=top_k)
            print(f"Top-{top_k} retrieved documents from vector search:", initial_docs)

            # 第二階段：使用 reranker 重排序
            nodes = [NodeWithScore(node=TextNode(text=doc.page_content)) for doc in initial_docs]
            query_bundle = QueryBundle(query_str=question)
            ranked_nodes = self.reranker._postprocess_nodes(nodes, query_bundle)

            # 取前 top_n_rerank 個節點並組合上下文
            top_ranked_nodes = ranked_nodes[:top_n_rerank]
            context = "\n".join([f"[text] {node.node.text}" for node in top_ranked_nodes])
            for idx, node in enumerate(top_ranked_nodes):
                print(f"Node {idx + 1}:\n{node.node.text}")
                print("\n" * 5)
            return context
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return None

    def generate_answer(self, context: str, question: str):
        if not context:
            return "No relevant context was found to answer the question."
        try:
            return self.answer_chain.run({'context': context, 'question': question})
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Failed to generate an answer."
    
    def run(self, question: str):
        context = self.retrieve_context(question)
        answer = self.generate_answer(context, question)
        return answer
