from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
import requests
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中读取 Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件是否配置正确！")

app = FastAPI()

# 初始化向量库
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hardware_datasheet")

class QueryRequest(BaseModel):
    question: str

def get_embedding(text):
    """直接调用阿里云百炼 API 生成向量（绕过 dashscope 库的 Bug）"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-v1",
        "input": {
            "texts": [text]
        }
    }
    try:
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()['output']['embeddings'][0]['embedding']
        else:
            print(f"API调用失败: {resp.text}")
            return None
    except Exception as e:
        print(f"请求异常: {e}")
        return None

@app.post("/ask")
async def ask_question(req: QueryRequest):
    # 1. 检索
    query_embedding = get_embedding(req.question)
    if query_embedding is None:
        return {"answer": "检索失败：无法获取问题的向量表示。"}
        
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context = "\n\n---\n\n".join(results['documents'][0])
    
    # 2. 生成（使用 OpenAI 兼容接口请求通义千问）
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "你是一个专业的 IT 硬件技术支持助手。请根据提供的文档回答问题。如果文档中没有相关信息，请直接回答“根据现有资料无法回答该问题”，不要编造。"},
            {"role": "user", "content": f"【参考文档】\n{context}\n\n【用户问题】\n{req.question}"}
        ]
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        answer = resp.json()['choices'][0]['message']['content']
    else:
        answer = f"大模型调用失败: {resp.text}"
        
    return {"answer": answer}