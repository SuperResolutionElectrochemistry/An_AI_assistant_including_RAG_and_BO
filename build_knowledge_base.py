
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


DATA_DIR  = Path("")
PAPER_DIR = Path("")
WORD_DIR  = Path("")
DB_DIR    = Path("")

print("DATA_DIR =", DATA_DIR, type(DATA_DIR))
print("PAPER_DIR =", PAPER_DIR, type(PAPER_DIR))
print("WORD_DIR  =", WORD_DIR, type(WORD_DIR))
print("DB_DIR =", DB_DIR, type(DB_DIR))


def load_pdfs():
    docs = []

    if not PAPER_DIR.exists():
        print(f"PDF 目录不存在：{PAPER_DIR}")
        return docs

    pdf_files = list(PAPER_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"在 {PAPER_DIR} 下没有找到 pdf 文件，请先放入 PDF 文献。")
        return docs

    for pdf_path in pdf_files:
        print(f"加载 PDF：{pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))  # 这里才转为字符串
        pdf_docs = loader.load()

        is_my_paper = pdf_path.name in MY_PAPERS

        for d in pdf_docs:
            d.metadata["source_type"] = "pdf"
            d.metadata["source_file"] = pdf_path.name
            d.metadata["is_my_paper"] = is_my_paper

        docs.extend(pdf_docs)

    print(f"共从 PDF 加载到 {len(docs)} 个文档块。")
    return docs


def load_csv():
    csv_path = ""

    if not csv_path.exists():
        print(f"未找到 CSV 文件：{csv_path}")
        return []

    print(f"加载 CSV：{csv_path.name}")
    loader = CSVLoader(
        file_path=str(csv_path),
        encoding="utf-8-sig"
    )

    csv_docs = loader.load()

    for d in csv_docs:
        d.metadata["source_type"] = "csv"
        d.metadata["source_file"] = csv_path.name

    print(f"从 CSV 读取到 {len(csv_docs)} 条文献数据。")
    return csv_docs


def load_word_docs():
    docs = []

    if not WORD_DIR.exists():
        print(f"Word 目录不存在：{WORD_DIR}")
        return docs

    docx_files = list(WORD_DIR.glob("*.docx"))
    if not docx_files:
        print(f"在 {WORD_DIR} 下没有找到任何 .docx 文件。")
        return docs

    for docx_path in docx_files:
        print(f"加载 Word 文档：{docx_path.name}")
        loader = Docx2txtLoader(str(docx_path))
        word_docs = loader.load()

        for d in word_docs:
            d.metadata["source_type"] = "word"
            d.metadata["source_file"] = docx_path.name

        docs.extend(word_docs)

    print(f"共从 Word 文档加载到 {len(docs)} 个文档块。")
    return docs


def load_element_stats():
    stats_path = ""
    print(f"{stats_path.name}")
    loader = CSVLoader(
        file_path=str(stats_path),
        encoding="utf-8-sig",
    )
    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "element_stats"
        d.metadata["source_file"] = stats_path.name

    return docs


def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    docs = splitter.split_documents(documents)
    print(f"切分后得到 {len(docs)} 个文档块。")
    return docs



def build_vector_store(docs):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )

    if not DB_DIR.exists():
        DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"开始构建向量库到：{DB_DIR}")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    vectorstore.persist()
    print("向量库构建完成！")
