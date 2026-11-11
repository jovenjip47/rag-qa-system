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
    {"q": "RAG 和 LoRA 微调分别改变什么？", "kw": ["推理时提供", "改变模型参数"]},
    {"q": "RAG 能保证模型绝不产生错误回答吗？", "kw": ["不能"]},
    {"q": "固定长度切块有什么缺点？", "kw": ["截断", "混合不同主题"]},
    {"q": "切块重叠会带来什么代价？", "kw": ["索引体积", "重复检索"]},
    {"q": "Top-K 在检索中是什么意思？", "kw": ["最多取回", "候选片段"]},
    {"q": "为什么不能把 Chroma 的距离分数直接叫相关度？", "kw": ["距离度量", "接口定义"]},
    {"q": "关键词检索适合什么内容？", "kw": ["名称", "编号", "术语"]},
    {"q": "正确证据检索到了但答案错，优先排查什么？", "kw": ["提示词", "上下文", "模型"]},
    {"q": "怎样避免关键词评分误判？", "kw": ["不完整", "错误回答", "关键词"]},
    {"q": "评估时如何避免测试集被用于调参？", "kw": ["验证集", "最终比较"]},
    {"q": "RAG 服务上线公网前应补哪些保护？", "kw": ["身份认证", "访问控制", "速率限制"]},
    {"q": "本项目 BGE 模型的向量维度是多少？", "kw": ["没有", "未提供", "资料不足", "未说明"]},
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


