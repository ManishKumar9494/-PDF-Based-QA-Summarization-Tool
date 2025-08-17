import streamlit as st
import pdfplumber
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------
#  Load small summarization model
# -----------------------
@st.cache_resource
def load_models():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return summarizer, embedding_model

summarizer, embedding_model = load_models()

# -----------------------
#  PDF Extraction
# -----------------------
def extract_pdf_text(pdf_file):
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Split by paragraph (or 500 chars if paragraph too long)
                paragraphs = [p for p in text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    for j in range(0, len(para), 500):
                        pages.append((i+1, para[j:j+500]))
    return pages

# -----------------------
#  Build FAISS index
# -----------------------
def build_faiss_index(pages):
    texts = [t for _, t in pages]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, texts, embeddings

# -----------------------
#  Structured Summarization
# -----------------------
def summarize_document_structured(pages):
    summaries = []
    for _, chunk in pages:
        # Remove code blocks or noise
        clean_chunk = re.sub(r"```.*?```", "", chunk, flags=re.DOTALL)
        clean_chunk = re.sub(r"\n\s*\n", "\n", clean_chunk)

        prompt = (
            f"You are an expert technical assistant. Summarize the following PDF content in a structured format:\n"
            f"- Use headings for each section\n"
            f"- Use bullet points for key tasks\n"
            f"- Include tools, technologies, or methods mentioned\n"
            f"- Avoid repeated or irrelevant text\n\n"
            f"Content:\n{clean_chunk}\n\nStructured Summary:"
        )
        out = summarizer(prompt)[0]["generated_text"]
        summaries.append(out)

    # Combine summaries into a final concise summary
    final_summary_text = " ".join(summaries)
    final_prompt = f"Combine the following chunk summaries into a concise, well-formatted structured summary:\n{final_summary_text}\n\nFinal Structured Summary:"
    final_summary = summarizer(final_prompt)[0]["generated_text"]
    return final_summary

# -----------------------
#  QA Retrieval + Generative Answer
# -----------------------
def answer_query(query, index, texts, embeddings, pages, top_k=3):
    #  Retrieve top relevant chunks
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    retrieved_chunks = [texts[i] for i in I[0]]
    cited_pages = [pages[i][0] for i in I[0]]

    #  Combine chunks into context
    context = " ".join(retrieved_chunks)

    #  Generate concise answer using Flan-T5
    prompt = f"Answer the question based only on the context below in a concise manner:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    answer = summarizer(prompt)[0]["generated_text"]

    return answer, cited_pages


st.title("Fast PDF QA & Structured Summarizer")

uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf:
    with st.spinner("Extracting PDF..."):
        pages = extract_pdf_text(uploaded_pdf)
    st.success(f"Extracted {len(pages)} chunks")

    with st.spinner("Building search index..."):
        index, texts, embeddings = build_faiss_index(pages)
    st.success("Index built ")

    if st.button("Generate Structured Summary"):
        with st.spinner("Generating structured summary..."):
            summary = summarize_document_structured(pages)
        st.subheader(" Document Structured Summary")
        st.write(summary)

    st.subheader(" Ask a Question")
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Finding concise answer..."):
            answer, citations = answer_query(query, index, texts, embeddings, pages)
        st.write("**Answer:**")
        st.write(answer)
        st.write(f" Cited Pages: {citations}")
