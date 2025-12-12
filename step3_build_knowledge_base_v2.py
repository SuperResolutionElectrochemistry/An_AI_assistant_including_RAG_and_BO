# 让 RAG 能看到 hea_element_stats.csv 的内容，通过load_element_stats加载"hea_element_stats.csv"，给自己发表的两篇文章打标签
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

MY_PAPERS = {
    "my-paper1-pan.pdf",
    "my-paper2-shan.pdf",
}

# ========= 1. 路径定义（全部封装成 Path，不要用 str） =========
DATA_DIR  = Path(r"E:\AMsystem\Project\QCH-RAG\data")
PAPER_DIR = Path(r"E:\AMsystem\Project\QCH-RAG\data\papers\pdf")
WORD_DIR  = Path(r"E:\AMsystem\Project\QCH-RAG\data\papers\word")
DB_DIR    = Path(r"E:\AMsystem\Project\QCH-RAG\db\8")

print("DATA_DIR =", DATA_DIR, type(DATA_DIR))
print("PAPER_DIR =", PAPER_DIR, type(PAPER_DIR))
print("WORD_DIR  =", WORD_DIR, type(WORD_DIR))
print("DB_DIR =", DB_DIR, type(DB_DIR))


# ========= 2. 加载 PDF 文献 =========
def load_pdfs():
    docs = []

    if not PAPER_DIR.exists():
        print(f"❌ PDF 目录不存在：{PAPER_DIR}")
        return docs

    pdf_files = list(PAPER_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️ 在 {PAPER_DIR} 下没有找到 pdf 文件，请先放入 PDF 文献。")
        return docs

    for pdf_path in pdf_files:
        print(f"加载 PDF：{pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))  # 这里才转为字符串
        pdf_docs = loader.load()

        # 🔥 根据文件名判断是不是“我的论文”
        is_my_paper = pdf_path.name in MY_PAPERS

        for d in pdf_docs:
            d.metadata["source_type"] = "pdf"
            d.metadata["source_file"] = pdf_path.name
            d.metadata["is_my_paper"] = is_my_paper  # 关键标记

        docs.extend(pdf_docs)

    print(f"共从 PDF 加载到 {len(docs)} 个文档块。")
    return docs


# ========= 3. 加载 CSV 数据 =========
def load_csv():
    csv_path = DATA_DIR / "her_hea_literature_clean.csv"

    if not csv_path.exists():
        print(f"❌ 未找到 CSV 文件：{csv_path}")
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


# ========= 4. 加载 Word（.docx）仪器/设备说明 =========
def load_word_docs():
    docs = []

    if not WORD_DIR.exists():
        print(f"⚠️ Word 目录不存在：{WORD_DIR}（如果有仪器说明 Word，请先创建该目录并放入 .docx 文件）")
        return docs

    docx_files = list(WORD_DIR.glob("*.docx"))
    if not docx_files:
        print(f"⚠️ 在 {WORD_DIR} 下没有找到任何 .docx 文件。")
        return docs

    for docx_path in docx_files:
        print(f"加载 Word 文档：{docx_path.name}")
        loader = Docx2txtLoader(str(docx_path))
        word_docs = loader.load()

        for d in word_docs:
            d.metadata["source_type"] = "word"
            d.metadata["source_file"] = docx_path.name

        docs.extend(word_docs)

    print(f"✅ 共从 Word 文档加载到 {len(docs)} 个文档块。")
    return docs


def load_element_stats():
    """加载 hea_element_stats.csv，供 RAG 使用"""
    stats_path = DATA_DIR / "hea_element_stats.csv"
    if not stats_path.exists():
        print(f"⚠️ 未找到元素统计表：{stats_path}")
        return []

    print(f"加载元素统计表：{stats_path.name}")
    loader = CSVLoader(
        file_path=str(stats_path),
        encoding="utf-8-sig",
    )
    docs = loader.load()

    for d in docs:
        d.metadata["source_type"] = "element_stats"
        d.metadata["source_file"] = stats_path.name

    print(f"✅ 从元素统计表读取到 {len(docs)} 条记录。")
    return docs


# ========= 5. 文本切割 =========
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # 1000
        chunk_overlap=200,  # 200
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    docs = splitter.split_documents(documents)
    print(f"切分后得到 {len(docs)} 个文档块。")
    return docs


# ========= 6. 构建向量库 =========
def build_vector_store(docs):
    # if "OPENAI_API_KEY" not in os.environ:
    #     print("❌ 未检测到 OPENAI_API_KEY，请先设置系统环境变量。")
    #     return
    #
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    embedding_api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")

    if not embedding_api_key or not embedding_base_url:
        print("❌ 未检测到 OPENAI_EMBEDDING_API_KEY 或 OPENAI_EMBEDDING_BASE_URL，请在 .env 或系统环境变量中配置。")
        return

    # ⚠️ 这里的 model 名称要与你在对应平台开通的 embedding 模型一致
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",  # 如果你那边是别的名字，就改成对应的
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
    print("✅ 向量库构建完成！")


# ========= 7. 主程序 =========
def main():
    pdf_docs = load_pdfs()
    csv_docs = load_csv()
    word_docs = load_word_docs()
    stats_docs = load_element_stats()  # ✅ 元素统计表

    all_docs = pdf_docs + csv_docs + word_docs + stats_docs
    if not all_docs:
        print("⚠️ 没有任何文档用于构建 RAG 知识库，请检查 PDF / CSV / Word。")
        return

    split = split_docs(all_docs)
    build_vector_store(split)


if __name__ == "__main__":
    main()