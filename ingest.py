# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
第一步：把知识文档变成向量，存入向量数据库（Chroma）
运行方式：python ingest.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')  # 国内下载模型走镜像

os.environ.setdefault('HF_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))  # 模型缓存放到项目内

os.environ.setdefault('HF_HUB_OFFLINE', '1')             # 模型已本地缓存，禁止联网

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')   # 关闭 Chroma 遥测

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KNOWLEDGE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "knowledge"      # 知识文档目录
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")               # 向量数据库目录
EMBED_MODEL = "BAAI/bge-small-zh-v1.5" # 中文向量模型（首次运行自动下载）


def main():
    # 1. 读取所有 .txt 知识文档
    docs = []
    for f in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        loader = TextLoader(str(f), encoding="utf-8")
        docs.extend(loader.load())
        print(f"已读取文档：{f.name}")

    if not docs:
        print("knowledge 目录下没有 .txt 文档，请先放几份文档进去。")
        return

    # 2. 把长文本切成小块（chunk），方便检索
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"文本切分完成：共 {len(chunks)} 个文本块")

    # 3. 向量化并存入 Chroma
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"知识库构建完成，已保存到 {CHROMA_DIR}")


if __name__ == "__main__":
    main()





