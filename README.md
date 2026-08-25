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

16 道测试题（包含大模型不可能知道的私有知识），直接问答准确率 62.5%，RAG 问答 100%，提升 37.5 个百分点。详细记录见 `实验报告.csv`。

## 说明

- 中文向量模型首次运行会自动下载并缓存到项目 `models/` 目录（脚本已配置国内镜像 hf-mirror.com）
- `.env` 含密钥，已被 `.gitignore` 排除，请勿上传到公开仓库
