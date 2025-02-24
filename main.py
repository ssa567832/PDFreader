from pipeline.preprocessing_pipeline import PreprocessingPipeline
from pipeline.qa_pipeline import QAPipeline
from managers.llm_manager import LLMManager
from managers.embedding_manager import EmbeddingManager
from managers.reranker_manager import RerankerManager

def main():
    # 基本配置
    pdf_path = "crowd.pdf"
    db_path = "db"
    api_base = "http://10.5.61.81:11437"
    llm_model_name = "cwchang/llama-3-taiwan-8b-instruct:f16"
    
    # 先初始化嵌入模型（共享給前處理與問答）
    em_model_name = "intfloat/multilingual-e5-large"
    model_kwargs = {'device': 'cpu'}
    embedding_manager = EmbeddingManager(model_name=em_model_name, model_kwargs=model_kwargs)
    embedding_function = embedding_manager.initialize_embeddings()
    
    # 執行前處理，更新 FAISS 向量庫
    preprocessing = PreprocessingPipeline(pdf_path=pdf_path, db_path=db_path, embedding_function=embedding_function)
    preprocessing.run()
    
    # 從前處理管道中重用建立好的向量庫物件
    vectorstore = preprocessing.vector_store_manager.vectorstore
    if not vectorstore:
        print("Failed to load vectorstore. Please check the path or embedding function.")
        return

    # 初始化 LLM 與 Reranker
    llm_manager = LLMManager(api_base=api_base, model_name=llm_model_name)
    llm = llm_manager.initialize_llm()
    
    reranker_manager = RerankerManager(top_n=7, model="BAAI/bge-reranker-large", use_fp16=True)
    reranker = reranker_manager.initialize_reranker()
    
    # 問答管道，使用已建立的向量庫
    qa_pipeline = QAPipeline(llm, vectorstore, reranker)
    question = input("Please enter your question: ")
    answer = qa_pipeline.run(question)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
