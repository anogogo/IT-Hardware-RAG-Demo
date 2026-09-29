# IT 硬件技术支持智能问答系统 (RAG Demo)

## 📖 项目简介
本项目是一个基于大语言模型（LLM）和检索增强生成（RAG）技术的 IT 硬件技术支持问答系统。

针对技术支持人员面对海量硬件 Datasheet（如交换机、服务器手册）时检索效率低、大模型容易产生幻觉（瞎编参数）的痛点。本项目通过将非结构化 PDF 文档转化为向量知识库，结合通义千问大模型，实现了基于真实文档的精准参数问答，并支持精确溯源到文档页码。

## 🛠️ 技术栈
- **编程语言**：Python 3.10+ (推荐 3.11)
- **数据提取与清洗**：pdfplumber, LangChain
- **向量化与检索**：阿里云百炼 (TextEmbedding), ChromaDB
- **大语言模型**：通义千问 (Qwen-Plus)
- **后端框架**：FastAPI, Uvicorn
- **环境与配置**：python-dotenv

## 📂 项目结构
```text
RAG_Demo/
├── .env.example          # 环境变量示例文件
├── .gitignore            # Git 忽略规则
├── requirements.txt      # 项目依赖库清单
├── extract.py            # PDF 文本提取与滑动窗口切分
├── embed.py              # 调用 Embedding API 构建 ChromaDB 向量库
├── rag_qa.py             # 命令行版 RAG 问答引擎
├── api.py                # FastAPI 后端接口服务
├── chunks.jsonl          # 切分好的知识块数据 (运行 extract.py 后生成)
└── chroma_db/            # 本地向量数据库 (运行 embed.py 后生成)

🚀 快速开始
1. 安装依赖
bash
pip install -r requirements.txt
2. 配置 API Key
前往阿里云百炼控制台创建 API Key。

在项目根目录下复制 .env.example 文件，并重命名为 .env。

在 .env 文件中填入你真实的 API Key：

text
DASHSCOPE_API_KEY=sk-你的真实Key
3. 构建知识库
将你的硬件 Datasheet（PDF格式）放入项目根目录。

修改 extract.py 中的 PDF_FILE_NAME 变量，指向你的 PDF 文件。

运行以下命令进行数据提取、切分和向量化：

bash
python extract.py
python embed.py
4. 测试问答
命令行测试：

bash
python rag_qa.py
启动 FastAPI 后端服务：

bash
uvicorn api:app --reload --port 8000
启动后，浏览器访问 http://127.0.0.1:8000/docs 打开 Swagger UI，即可在网页上测试接口。

💡 项目亮点
精准溯源，拒绝幻觉：通过严格的 Prompt 约束，系统在回答时能精确引用文档页码（如“根据文档第10页”），对未知问题能诚实回答“根据现有资料无法回答”，绝不胡编乱造。

工程化落地：不仅实现了 RAG 的核心算法，还使用 FastAPI 封装了标准的 RESTful 接口，实现了从“数据清洗 -> 向量入库 -> 后端服务”的完整工程流程。

模块化设计：数据清洗、向量构建、检索生成、接口服务高度解耦，方便后续接入企业内部真实知识库。

安全的密钥管理：使用 python-dotenv 将 API Key 抽离到 .env 文件中，防止密钥在代码仓库中泄露。

📝 后续优化计划
□ 引入 Streamlit 开发更友好的可视化聊天交互界面。
□ 支持扫描版 PDF 的 OCR 文本识别。
□ 引入混合检索 (BM25 + 向量检索) 提升检索准确率。