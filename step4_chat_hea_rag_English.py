import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ✅ 注意：下面几个都从 langchain_core 来，而不是 langchain.chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


DB_DIR = Path(r"E:\AMsystem\Project\QCH-RAG\db\8")

print("DB_DIR =", DB_DIR, type(DB_DIR))


def load_vectorstore():
    """加载已经构建好的 Chroma 向量库"""
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ 未检测到 OPENAI_API_KEY，请先设置环境变量。")
        return None
    embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
    if not embedding_api_key or not embedding_base_url:
        print("❌ 未检测到 OPENAI_EMBEDDING_API_KEY 或 OPENAI_EMBEDDING_BASE_URL，请在 .env 或系统环境变量中配置。")
        return
    if not DB_DIR.exists():
        print(f"❌ 未找到向量库目录：{DB_DIR}，请先运行 build_knowledge_base.py 构建知识库。")
        return None

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",  # 如果你那边是别的名字，就改成对应的
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    return vectorstore


def build_rag_chain():
    """用 LCEL 手工搭一个 RAG 链，避免使用 RetrievalQA"""

    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


    # 读取聊天模型用的 key / base_url
    # 优先级：OPENAI_API_KEY > OPENAI_CHAT_API_KEY > OPENAI_EMBEDDING_API_KEY
    chat_api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_CHAT_API_KEY")
        or os.getenv("OPENAI_EMBEDDING_API_KEY")
    )
    chat_base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_CHAT_BASE_URL")
        or os.getenv("OPENAI_EMBEDDING_BASE_URL")
    )

    if chat_api_key is None:
        print("❌ 没有找到用于 ChatOpenAI 的 api_key，请设置：OPENAI_API_KEY 或 OPENAI_CHAT_API_KEY。")
        return None

    # 创建 LLM（显式传入 api_key / base_url）
    llm = ChatOpenAI(
        model="gpt-4o",   # 或 "gpt-4o-mini"
        temperature=0.2,
        api_key=chat_api_key,
        base_url=chat_base_url,   # 如果用官方 OpenAI，可以留空或不传
    )

    # Prompt：把 context + question 填进去
#     prompt = ChatPromptTemplate.from_template(
#         """
# 你是 Chatalyst，是一名专注于高熵合金（HEA）和析氢反应（HER）的科研助手。
# 你可以访问到的“上下文”来自以下几类来源：
# - 高熵合金/HER 相关文献的 PDF
# - 我整理的文献数据表（包括元素组合、过电位、塔菲尔斜率等）
# - 我自己发表的相关文章（实验方法和流程）
#
# 请严格遵守以下规则回答：
# 1. 尽量基于给定的上下文（context）作答，不要凭空杜撰文献中的具体数字。
# 2. 如果上下文中没有直接答案，请明确说明“在当前知识库中没有直接信息”，然后再给出一般性的推断。
# 3. 回答时用英文，结构化表达（可以分点列出）。
# 4. 当用户让你“设计实验方案”时，优先模仿我自己文章中的实验方法和参数风格。
#
# --------------------
# 【检索到的相关内容（context）】：
# {context}
# --------------------
# 【用户问题】：
# {question}
# --------------------
# 请给出你的专业解答：
# """
#     )
    prompt = ChatPromptTemplate.from_template(
        """
You are Chatalyst, a research assistant specializing in high-entropy alloys (HEAs) and the hydrogen evolution reaction (HER).
You have access to the following types of context:
- PDF literature related to HEAs and HER
- My curated literature tables (elemental combinations, overpotential, Tafel slope, etc.)
- My own published papers (experimental methods and workflows)

Please strictly follow these rules:
1. Base your answers on the provided context whenever possible. Do not fabricate numerical data that does not appear in the context.
2. If the context does not contain a direct answer, explicitly state: “The current knowledge base does not contain direct information,” and then provide a general, well-reasoned explanation.
3. Answer **in English**, using a clear **structured format** (bullet points or numbered lists).
4. When asked to **design an experimental procedure**, prioritize mimicking the style, structure, and parameter selection found in my own published papers.

-----------------------------------------------------
【Important rule when numbered items appear in context】
If the retrieved context contains **numbered definitions or enumerated lists**  
(e.g., “1. … 2. … 3. …” or similar structures), Chatalyst must:

- Detect the presence of an enumerated list.
- **Always output the complete set of items**, even if the context only includes a partial list due to chunking or retrieval limits.
- If only part of the list is present in the context, supplement the missing items using established electrochemical knowledge (e.g., ECSA, Faradaic efficiency, current density, stability, TOF for HEA electrocatalysis).
- Ensure that numbering is continuous and complete (e.g., 1–8), with no missing or repeated entries.

This rule ensures that definitions, terminology lists, or concept summaries are always **complete**, even if the retrieved context includes only partial chunks.

-----------------------------------------------------
【Retrieved Context (context)】:
{context}

【User Question】:
{question}

-----------------------------------------------------
Please provide your expert answer:
"""
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



def chat_loop():
    rag_chain = build_rag_chain()
    if rag_chain is None:
        return

    print("✅ RAG 助手已启动。输入你的问题，输入 q 退出。\n")

    while True:
        question = input("Human input：").strip()
        if not question:
            continue
        if question.lower() in ["q", "quit", "exit"]:
            print("再见！")
            break

        # 调用链：把 question 作为输入
        try:
            answer = rag_chain.invoke(question)
        except Exception as e:
            print("❌ 调用 RAG 链出错：", e)
            continue

        print("\nAI Assistant：", answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    chat_loop()
