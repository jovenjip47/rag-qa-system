"""检索评测：python evaluate.py；输出供人工复核的 CSV，不自动判定回答正确。"""
import csv
import json
import os
import argparse
from pathlib import Path

from query import load_vectorstore, build_prompt

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="检索评测；可选调用模型生成待人工评分的回答")
    parser.add_argument("--generate", action="store_true", help="调用 DeepSeek API，会产生费用")
    args = parser.parse_args()
    client = None
    if args.generate:
        from openai import OpenAI
        if not os.getenv("DEEPSEEK_API_KEY"):
            parser.error("请先配置 DEEPSEEK_API_KEY")
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    cases = json.loads((BASE / "evaluation_cases.json").read_text(encoding="utf-8"))
    store = load_vectorstore()
    rows = []
    for case in cases:
        results = store.similarity_search_with_score(case["question"], k=4)
        sources = [Path(doc.metadata.get("source", "").replace("\\", "/")).name for doc, _ in results]
        expected = case["source"]
        hit = expected in sources if expected else None
        answer = ""
        if client:
            context = "\n\n".join(f"[资料{i+1}] {doc.page_content}" for i, (doc, _) in enumerate(results))
            response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": build_prompt(case["question"], context)}], temperature=0.3)
            answer = response.choices[0].message.content or ""
        rows.append({"question": case["question"], "reference": case["reference"],
                     "expected_source": expected or "无答案", "retrieval_hit": "待人工判断" if hit is None else int(hit),
                     "retrieved_sources": " | ".join(sources),
                     "retrieved_chunks": "\n---\n".join(doc.page_content for doc, _ in results),
                     "answer": answer, "answer_correct_manual": "", "grounded_manual": "", "notes": ""})
    path = BASE / "evaluation_results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    eligible = [r for r in rows if isinstance(r["retrieval_hit"], int)]
    print(f"来源命中 {sum(r['retrieval_hit'] for r in eligible)}/{len(eligible)}；结果保存在 {path}")
    print("来源命中不代表片段包含答案，也不代表回答正确；请逐条检查片段。")


if __name__ == "__main__":
    main()
