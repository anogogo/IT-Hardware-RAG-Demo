import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

# ⚠️⚠️⚠️ 注意！这行必须改成你左侧PDF的真实完整名字！⚠️⚠️⚠️
PDF_FILE_NAME = "H3C S5560X & S6520X 系列以太网交换机 NetStream技术白皮书-6W101-整本手册.pdf" 

def extract_text_from_pdf(pdf_path):
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_content.append(f"--- 第 {i+1} 页 ---\n{text}")
    return "\n".join(text_content)

if __name__ == "__main__":
    print("开始提取PDF...")
    raw_text = extract_text_from_pdf(PDF_FILE_NAME)
    
    with open("raw_datasheet.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"提取完成！共提取 {len(raw_text)} 个字符。")

    print("开始切分文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,       
        chunk_overlap=50,     
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""] 
    )
    chunks = text_splitter.split_text(raw_text)
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        formatted_chunks.append({
            "id": i,
            "text": chunk,
            "source": PDF_FILE_NAME
        })

    with open("chunks.jsonl", "w", encoding="utf-8") as f:
        for item in formatted_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"切分完成！共生成 {len(chunks)} 个文本块。")