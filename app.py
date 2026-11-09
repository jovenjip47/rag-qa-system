# -*- coding: utf-8 -*-
"""
第四步：把 RAG 问答做成网页（FastAPI）
运行方式：python app.py
然后浏览器打开 http://127.0.0.1:8000
停止服务：在运行窗口按 Ctrl + C
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
TOP_K = 4

# 启动时加载向量库和模型（第一次启动需要 20~30 秒）
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

app = FastAPI(title="RAG 知识库问答系统")


def build_prompt(question, context):
    return f"""你是一个知识库问答助手。请只根据下面提供的【资料】回答用户的问题。
如果资料里没有相关内容，请直接说"资料中没有相关信息"，不要自己编造。

【资料】
{context}

【问题】
{question}

请给出简洁、准确的回答："""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE


@app.post("/ask")
def ask(question: str = Form(...)):
    results = vectorstore.similarity_search_with_score(question, k=TOP_K)
    context = "\n\n".join(f"[资料{i+1}] {d.page_content}" for i, (d, _) in enumerate(results))
    prompt = build_prompt(question, context)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content.strip()
    sources = [
        {"index": i + 1, "content": d.page_content, "source": d.metadata.get("source", "未知"), "distance": round(float(s), 3)}
        for i, (d, s) in enumerate(results)
    ]
    return JSONResponse({"question": question, "answer": answer, "sources": sources})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".txt"):
        return JSONResponse({"ok": False, "message": "目前只支持 .txt 文本文件"}, status_code=400)
    content = (await file.read()).decode("utf-8", errors="replace")
    chunks = splitter.split_text(content)
    docs = [Document(page_content=c, metadata={"source": file.filename}) for c in chunks]
    if not docs:
        return JSONResponse({"ok": False, "message": "文件没有可索引的内容"}, status_code=400)
    vectorstore.add_documents(docs)
    return JSONResponse({"ok": True, "message": f"上传成功，已把 {len(chunks)} 个文本块加入知识库，现在可以直接提问了"})


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>RAG 知识库问答系统</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; background: #f7f8fa; color: #333; }
  h1 { font-size: 22px; }
  .card { background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  textarea { width: 100%; height: 60px; border: 1px solid #ddd; border-radius: 6px; padding: 8px; font-size: 14px; box-sizing: border-box; }
  button { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 8px 18px; cursor: pointer; font-size: 14px; }
  button:hover { background: #1d4ed8; }
  .ans { white-space: pre-wrap; line-height: 1.8; }
  .src { background: #f3f4f6; border-radius: 6px; padding: 10px; margin-top: 8px; font-size: 13px; line-height: 1.6; }
  .score { color: #999; }
  .status { color: #999; margin-left: 10px; }
</style>
</head>
<body>
  <h1>📚 RAG 知识库问答系统</h1>
  <div class="card">
    <textarea id="q" placeholder="输入你的问题，例如：云帆科技成立于哪一年？"></textarea>
    <br><br>
    <button onclick="ask()">提问</button>
    <span class="status" id="status"></span>
  </div>
  <div class="card" id="answerCard" style="display:none">
    <div class="ans" id="answer"></div>
    <hr>
    <div id="sources"></div>
  </div>
  <div class="card">
    <h3>上传文档（扩充知识库，支持 .txt）</h3>
    <input type="file" id="file" accept=".txt">
    <button onclick="upload()">上传</button>
    <span class="status" id="upStatus"></span>
  </div>
  <script>
    async function ask() {
      const q = document.getElementById("q").value.trim();
      if (!q) { alert("请输入问题"); return; }
      document.getElementById("status").textContent = "思考中…";
      document.getElementById("answer").textContent = "";
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: "question=" + encodeURIComponent(q)
      });
      const data = await res.json();
      document.getElementById("status").textContent = "";
      document.getElementById("answer").textContent = data.answer;
      const sources = document.getElementById("sources");
      sources.replaceChildren();
      sources.append(document.createTextNode("检索到的资料："));
      for (const s of data.sources) {
        const div = document.createElement("div");
        div.className = "src";
        div.textContent = `[${s.index}] ${s.source}（距离 ${s.distance}）\n${s.content}`;
        sources.append(div);
      }
      document.getElementById("answerCard").style.display = "block";
    }
    async function upload() {
      const f = document.getElementById("file").files[0];
      if (!f) { alert("请选择文件"); return; }
      const fd = new FormData();
      fd.append("file", f);
      document.getElementById("upStatus").textContent = "上传中…";
      const res = await fetch("/upload", { method: "POST", body: fd });
      const data = await res.json();
      document.getElementById("upStatus").textContent = data.message;
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("启动中，浏览器打开 http://127.0.0.1:8000 （首次启动需加载模型，约 20~30 秒）")
    uvicorn.run(app, host="127.0.0.1", port=8000)


