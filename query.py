# -*- coding: utf-8 -*-
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
"""
第二步：提问 → 检索知识库 → 拼接 Prompt → 调用 DeepSeek 回答
运行方式：python query.py "你的问题"
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')  # 国内下载模型走镜像

os.environ.setdefault('HF_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))  # 模型缓存放到项目内



os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')   # 关闭 Chroma 遥测

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 4  # 每次返回最相关的 4 段资料


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def build_prompt(question: str, context: str) -> str:
    return f"""你是一个知识库问答助手。请只根据下面提供的【资料】回答用户的问题。
如果资料里没有相关内容，请直接说"资料中没有相关信息"，不要自己编造。

【资料】
{context}

【问题】
{question}

请给出简洁、准确的回答："""


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "什么是 RAG？"

    # 1. 加载知识库
    vectorstore = load_vectorstore()

    # 2. 检索最相关的文本
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    context = "\n\n".join(
        f"[资料{i+1}] {doc.page_content}" for i, (doc, _score) in enumerate(results)
    )

    # 3. 拼接 Prompt
    prompt = build_prompt(question, context)

    # 4. 调用 DeepSeek 大模型
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content

    # 5. 打印结果
    print("=" * 50)
    print(f"【问题】{question}")
    print("【回答】")
    print(answer)
    print("=" * 50)
    print("【回答依据（检索到的资料）】")
    for i, (doc, score) in enumerate(results):
        print(f"[{i+1}] 来源 {doc.metadata.get('source', '未知')}，距离 {score:.3f}：{doc.page_content[:60]}...")


if __name__ == "__main__":
    main()






