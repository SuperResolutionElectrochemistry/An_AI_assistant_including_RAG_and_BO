import os
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DB_DIR = Path("")

print("DB_DIR =", DB_DIR, type(DB_DIR))


def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )
    return vectorstore


def build_rag_chain():
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # 创建 LLM
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.2,
        api_key=chat_api_key,
        base_url=chat_base_url,
    )

    
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

    while True:
        question = input("Human input：").strip()
        if not question:
            continue
        if question.lower() in ["q", "quit", "exit"]:
            print("再见！")
            break

        try:
            answer = rag_chain.invoke(question)
        except Exception as e:
            print("调用 RAG 链出错：", e)
            continue

        print("\nAI Assistant：", answer)
        print("\n" + "-" * 60 + "\n")


