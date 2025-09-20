import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from src.config import AppConfig
from src.chunker import chunk_texts
from src.vectorstore import VectorStore
from src.rag import build_prompt, format_sources
from src.llm import LLMClient
from src.pdf_utils import extract_pages_text

load_dotenv()
cfg = AppConfig.from_env()

st.set_page_config(page_title="AI Marketplace Assistant", page_icon="🛍️", layout="wide")
st.sidebar.title("⚙️ Settings")

question = st.sidebar.text_area("Ask a question about a product", height=100)
ask_btn = st.sidebar.button("Ask")

st.title("🛍️ AI-Powered Marketplace Assistant for Local Artisans")

uploaded_files = st.file_uploader(
    "Upload product catalog (CSV or PDF)", 
    accept_multiple_files=True, 
    type=["csv", "pdf"]
)

uploads_dir = Path("data/uploads")
index_dir = Path("data/index")
uploads_dir.mkdir(parents=True, exist_ok=True)
index_dir.mkdir(parents=True, exist_ok=True)

if uploaded_files:
    for f in uploaded_files:
        (uploads_dir / f.name).write_bytes(f.read())
    st.success(f"Uploaded {len(uploaded_files)} file(s)")

if st.button("Build / Update Product Index"):
    all_products = []

    # CSV Processing
    csv_files = list(uploads_dir.glob("*.csv"))
    for csv_path in csv_files:
        st.text(f"Processing {csv_path.name}...")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            all_products.append({
                "source": str(csv_path),
                "product_name": row.get("product_name", ""),
                "description": row.get("description", ""),
                "text": f"{row.get('product_name', '')} {row.get('description', '')}".strip()
            })

    # PDF Processing
    pdf_files = list(uploads_dir.glob("*.pdf"))
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, pdf_path in enumerate(tqdm(pdf_files, desc="Extracting text from PDFs")):
        status_text.text(f"Extracting text from {pdf_path.name}...")
        pages = extract_pages_text(str(pdf_path))
        for page_num, text in pages:
            all_products.append({
                "source": str(pdf_path),
                "product_name": "",
                "description": text.strip(),
                "text": text.strip()
            })
        progress_bar.progress((i + 1) / max(len(pdf_files), 1))

    if all_products:
        status_text.text("Chunking and embedding product data...")
        texts = [item['text'] for item in all_products]
        metas = [{"source": item['source'], "product_name": item['product_name']} for item in all_products]

        chunks, chunk_metas = chunk_texts(texts, metas, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        vs = VectorStore(str(index_dir), cfg.OPENAI_EMBED_MODEL, cfg.OPENAI_API_KEY)
        vs.build_or_update(chunks, chunk_metas)

        progress_bar.progress(1.0)
        status_text.text("Product index built successfully!")
        st.success("Product index built!")
    else:
        status_text.text("No product data found to index.")
        st.warning("Upload CSV or PDF files containing product info.")

if ask_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            vs = VectorStore(str(index_dir), cfg.OPENAI_EMBED_MODEL, cfg.OPENAI_API_KEY)
            if not os.path.exists(vs.index_path):
                st.error("Vector index not found. Please upload product data and build the index first.")
            else:
                results = vs.search(question, top_k=cfg.TOP_K)
                system_prompt, user_prompt = build_prompt(question, results, cfg.SYSTEM_PROMPT)
                llm = LLMClient(cfg.OPENAI_API_KEY, cfg.OPENAI_CHAT_MODEL)

                st.subheader("Answer")
                answer_placeholder = st.empty()
                answer = ""
                for delta in llm.stream_chat(system_prompt, user_prompt):
                    answer += delta
                    answer_placeholder.code(answer, language="markdown")

                st.subheader("Relevant Products")
                if results:
                    st.markdown(format_sources(results))
                else:
                    st.info("No matching products found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Please check your API key and ensure the index is built.")
