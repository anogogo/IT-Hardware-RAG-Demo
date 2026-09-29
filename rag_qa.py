import json
import chromadb
from dashscope import TextEmbedding, Generation
import dashscope
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中读取 Key
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("未找到 DASHSCOPE_API_KEY，请检查 .env 文件是否配置正确！")

# 设置 dashscope 的 API Key
dashscope.api_key = DASHSCOPE_API_KEY

# 初始化向量数据库客户端
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="hardware_datasheet")

def get_embedding(text):
    """把用户问题转为向量"""
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text
    )
    if resp.status_code == 200:
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"Embedding API调用失败: {resp}")
        return None

def retrieve_context(query, top_k=3):
    """在向量数据库中检索最相关的 top_k 个文本块"""
    query_embedding = get_embedding(query)
    if not query_embedding:
        return ""
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 拼接检索到的文本内容
    retrieved_texts = results['documents'][0]
    context = "\n\n---\n\n".join(retrieved_texts)
    return context

def ask_llm(query, context):
    """调用通义千问大模型，结合上下文回答问题"""
    prompt = f"""你是一个专业的 IT 硬件技术支持助手。请根据以下提供的技术文档片段，回答用户的问题。
如果文档中没有相关信息，请直接回答“根据现有资料无法回答该问题”，不要编造。

【参考文档】
{context}

【用户问题】
{query}

【回答】"""
    
    resp = Generation.call(
        model="qwen-plus",  # 使用通义千问 plus 模型
        prompt=prompt,
        result_format='message'
    )
    
    if resp.status_code == 200:
        return resp.output.choices[0].message.content
    else:
        return f"大模型调用失败: {resp}"

if __name__ == "__main__":
    print("🤖 RAG 智能问答系统已启动（输入 '退出' 结束）\n")
    while True:
        query = input("请输入你的问题: ")
        if query in ["退出", "exit", "quit"]:
            break
        
        print("正在检索知识库...")
        context = retrieve_context(query)
        
        print("正在生成回答...\n")
        answer = ask_llm(query, context)
        print(f"回答: {answer}\n")
        print("-" * 50)