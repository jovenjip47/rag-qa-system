# -*- coding: utf-8 -*-
"""
第三步：对比实验 —— 直接问大模型 vs RAG 检索后回答
运行方式：python compare.py
结果会保存到 实验报告.csv，并打印统计结论
"""
import sys, os, csv
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 3

# 测试题：kw 是"答案里应该出现的关键词"，命中任意一个即算答对
TEST_QUESTIONS = [
    {"q": "RAG 是什么的缩写？", "kw": ["检索增强生成", "Retrieval"]},
    {"q": "RAG 的完整流程一般包括哪些步骤？", "kw": ["切分", "向量", "检索", "Prompt"]},
    {"q": "RAG 相比直接问答有什么好处？", "kw": ["依据", "幻觉"]},
    {"q": "RAG 中向量数据库用来做什么？", "kw": ["向量", "存储"]},
    {"q": "Python 中列表用什么括号表示？", "kw": ["方括号", "[]"]},
    {"q": "Python 中元组创建后可以修改吗？", "kw": ["不可", "不能", "无法"]},
    {"q": "Python 中字典存放什么？", "kw": ["键值", "key-value"]},
    {"q": "Hadoop 主要包含哪两部分？", "kw": ["HDFS", "MapReduce"]},
    {"q": "Hive 是什么工具？", "kw": ["数据仓库"]},
    {"q": "HDFS 负责什么？", "kw": ["存储", "文件"]},
    {"q": "云帆科技成立于哪一年？", "kw": ["2016"]},
    {"q": "云帆科技总部在哪里？", "kw": ["深圳"]},
    {"q": "云帆科技有多少名员工？", "kw": ["200"]},
    {"q": "云帆科技 2025 年推出了什么新功能？", "kw": ["AI 客服"]},
    {"q": "云帆科技主要做什么产品？", "kw": ["知识库"]},
    {"q": "云帆科技的企业价值观是什么？", "kw": ["客户第一"]},
]


def get_client():
    return OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")


def ask_model(client, messages):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def direct_answer(client, question):
    return ask_model(client, [{"role": "user", "content": f"请回答下面的问题（如果不知道就直接说不知道）：\n{question}"}])


def rag_answer(client, question, vectorstore):
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    context = "\n\n".join(f"[资料{i+1}] {doc.page_content}" for i, (doc, _) in enumerate(results))
    prompt = f"""你是一个知识库问答助手。请只根据下面提供的【资料】回答用户的问题。
如果资料里没有相关内容，请直接说"资料中没有相关信息"，不要自己编造。

【资料】
{context}

【问题】
{question}

请给出简洁、准确的回答："""
    return ask_model(client, [{"role": "user", "content": prompt}])


def hit(answer, keywords):
    return any(k.lower() in answer.lower() for k in keywords)


def main():
    client = get_client()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    rows = []
    direct_ok = 0
    rag_ok = 0
    total = len(TEST_QUESTIONS)
    print(f"共 {total} 道测试题，开始对比...\n")

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["q"]
        print(f"[{i}/{total}] {q}")
        d = direct_answer(client, q)
        r = rag_answer(client, q, vectorstore)
        dh = hit(d, item["kw"])
        rh = hit(r, item["kw"])
        if dh:
            direct_ok += 1
        if rh:
            rag_ok += 1
        rows.append([q, d, "命中" if dh else "未命中", r, "命中" if rh else "未命中"])
        print(f"    直接问答: {'命中' if dh else '未命中'} | RAG: {'命中' if rh else '未命中'}")

    print("\n" + "=" * 50)
    print(f"直接问答 命中：{direct_ok}/{total}（{direct_ok / total * 100:.1f}%）")
    print(f"RAG 问答 命中：{rag_ok}/{total}（{rag_ok / total * 100:.1f}%）")
    print(f"关键词命中差异：{rag_ok - direct_ok:+d} 题（{(rag_ok - direct_ok) / total * 100:+.1f} 个百分点）；命中不等于回答正确，请人工复核。")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "实验报告.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["问题", "直接问答答案", "直接问答是否命中", "RAG答案", "RAG是否命中"])
        w.writerows(rows)
    print("\n详细结果已保存到：实验报告.csv（可用 Excel 打开）")


if __name__ == "__main__":
    main()


