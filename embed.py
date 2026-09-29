import json
import chromadb
from dashscope import TextEmbedding
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

def get_embedding(text):
    """调用阿里云百炼API将文本转为向量"""
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text
    )
    if resp.status_code == 200:
        return resp.output['embeddings'][0]['embedding']
    else:
        print(f"API调用失败: {resp}")
        return None

if __name__ == "__main__":
    print("正在加载清洗好的数据...")
    chunks = []
    with open("chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"共加载 {len(chunks)} 个文本块。")

    # 初始化本地向量数据库 Chroma
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection(name="hardware_datasheet")
    except:
        pass
    collection = client.create_collection(name="hardware_datasheet")

    print("开始生成词向量并写入数据库（请耐心等待）...")
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk['text'])
        if embedding:
            collection.add(
                embeddings=[embedding],
                documents=[chunk['text']],
                metadatas=[{"source": chunk['source'], "id": chunk['id']}],
                ids=[str(chunk['id'])]
            )
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(chunks)} 个文本块...")

    print("🎉 向量数据库构建完成！已保存至 ./chroma_db 文件夹。")