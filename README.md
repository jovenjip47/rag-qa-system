# RAG 知识库问答系统

基于检索增强生成（RAG）的中文知识库问答系统。支持命令行问答、对比实验和网页版（提问 / 查看引用来源 / 上传文档）。

## 功能

- **知识库构建**：读取 txt 文档 → 文本切分 → 向量化 → 存入 Chroma
- **命令行问答**：提问 → 检索 → 拼接 Prompt → DeepSeek 大模型生成回答
- **对比实验**：直接问答 vs RAG 问答，自动打分并输出 CSV 报告
- **网页版**：FastAPI 封装接口，支持提问、查看引用来源、上传 txt 文档扩充知识库

## 技术栈

Python · LangChain · Chroma · BGE（bge-small-zh-v1.5）· DeepSeek API · FastAPI

## 目录结构

```
rag_project/
├── ingest.py        # 构建知识库（切分→向量化→存 Chroma）
├── query.py         # 命令行问答（检索→Prompt→生成）
├── compare.py       # 对比实验（直接问答 vs RAG 问答）
├── app.py           # 网页版（FastAPI）
├── knowledge/       # 知识文档（txt）
├── requirements.txt # 依赖清单
├── .env.example     # 密钥配置模板
└── 实验报告.csv      # 对比实验详细结果
```

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 配置密钥：复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key
3. 构建知识库：
   ```bash
   python ingest.py
   ```
4. 命令行问答：
   ```bash
   python query.py "什么是 RAG？"
   ```
5. 对比实验：
   ```bash
   python compare.py
   ```
6. 网页版：
   ```bash
   python app.py
   # 浏览器打开 http://127.0.0.1:8000
   ```

## 实验效果

历史报告 `实验报告.csv` 使用任意关键词命中规则，存在把未回答判为命中的情况；其中数字是关键词命中率，不是准确率，也不能证明实际提升。

## 评测与限制

运行 `python evaluate.py` 检查 20 条示例题的前 4 个检索来源，结果写入 `evaluation_results.csv`。运行 `python evaluate.py --generate` 还会调用 DeepSeek 并产生 API 费用，把回答写入 CSV。请人工填写回答正确性和是否有资料依据；无答案题需人工检查是否合理拒答。文件名命中只是粗粒度指标，同一文件的错误片段也会计为命中，不能代替片段和回答检查。这些题基于示例语料编写，只适合练习，不代表真实场景性能。

首次运行需要联网下载 BGE 模型；如果已有缓存且需要离线使用，可自行设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`。安装 PyTorch 时请依据自己的 CPU 或 CUDA 环境选择版本。重跑 `ingest.py` 会重建向量库，清除此前通过网页上传的内容；网页重复上传目前仍会产生重复片段。

本项目为本地演示，调用 DeepSeek 时问题及检索内容会发送到 API。运行前请勿将敏感文档上传。未提供认证，不应直接暴露到公网。

## 说明

- 中文向量模型首次运行会自动下载并缓存到项目 `models/` 目录（脚本已配置国内镜像 hf-mirror.com）
- `.env` 含密钥，已被 `.gitignore` 排除，请勿上传到公开仓库

