import streamlit as st
from managers.llm_manager import LLMManager
from managers.embedding_manager import EmbeddingManager
from managers.reranker_manager import RerankerManager
from managers.vectorstore_manager import VectorStoreManager
from pipeline.qa_pipeline import QAPipeline

# ---- 配置 ----
DB_PATH = "db"
API_BASE = "http://10.5.61.81:11437"
LLM_MODEL = "cwchang/llama-3-taiwan-8b-instruct:f16"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# ---- Streamlit 標題 ----
st.title("📚 AI 問答系統")
st.markdown("**請輸入您的問題，系統會根據 PDF 內容提供回答**")

@st.cache_resource
def load_faiss_vectorstore():
    """載入 FAISS 向量庫"""
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    embedding_function = embedding_manager.initialize_embeddings()

    vector_store_manager = VectorStoreManager(DB_PATH, embedding_function)
    return vector_store_manager.vectorstore

@st.cache_resource
def initialize_llm():
    """初始化 LLM"""
    llm_manager = LLMManager(api_base=API_BASE, model_name=LLM_MODEL)
    return llm_manager.initialize_llm()

@st.cache_resource
def initialize_reranker():
    """初始化 Reranker"""
    reranker_manager = RerankerManager(top_n=7, model=RERANKER_MODEL, use_fp16=False)
    return reranker_manager.initialize_reranker()

# 載入模型
vectorstore = load_faiss_vectorstore()
llm = initialize_llm()
reranker = initialize_reranker()

# 如果 FAISS 載入失敗，顯示錯誤
if not vectorstore:
    st.error("❌ 無法載入 FAISS 向量庫，請檢查 db/index.faiss 是否存在")
    st.stop()

# 初始化問答管道
qa_pipeline = QAPipeline(llm, vectorstore, reranker)

# ---- 使用者輸入區域 ----
question = st.text_input("🔍 請輸入您的問題")

if st.button("問 AI"):
    if question.strip() == "":
        st.warning("請輸入有效的問題！")
    else:
        with st.spinner("⏳ AI 正在思考中..."):
            answer = qa_pipeline.run(question)
        st.success("✅ AI 回答完成！")
        st.write("**答案：**")
        st.info(answer)

# ---- 顯示 FAISS 向量庫的內容 ----
if st.sidebar.checkbox("📌 顯示向量庫資訊"):
    st.sidebar.subheader("向量庫資訊")
    st.sidebar.write(f"向量庫大小: {vectorstore.index.ntotal} 條資料")
