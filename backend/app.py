import os
import streamlit as st
import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
import Pinecone

# ------------------ PAGE ------------------
st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

# ------------------ ENV CHECK ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

st.write("GROQ_API_KEY present:", bool(GROQ_API_KEY))
st.write("COHERE_API_KEY present:", bool(COHERE_API_KEY))
st.write("PINECONE_API_KEY present:", bool(PINECONE_API_KEY))
st.write("PINECONE_INDEX_NAME present:", bool(PINECONE_INDEX_NAME))

if not all([GROQ_API_KEY, COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("Missing one or more required environment variables.")
    st.stop()

# ------------------ CLIENTS ------------------
co = cohere.Client(COHERE_API_KEY)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Pinecone init (safe)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="gcp-starter"  # works for free tier
)

# ------------------ SESSION ------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "has_data" not in st.session_state:
    st.session_state.has_data = False

# ------------------ RERANK ------------------
def rerank_docs(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]

    results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n,
    )

    return [docs[r.index] for r in results.results]

# ------------------ INGEST ------------------
st.subheader("Ingest Document")
text = st.text_area("Paste text to ingest", height=200)

if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    docs = splitter.create_documents([text])

    texts = []
    metadatas = []

    for i, doc in enumerate(docs):
        texts.append(doc.page_content)
        metadatas.append({
            "source": "user_input",
            "chunk_id": i,
        })

    # SAFE Pinecone ingestion
    vectorstore = LangChainPinecone.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        index_name=PINECONE_INDEX_NAME,
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.has_data = True

    st.success(f"Ingested {len(texts)} chunks into Pinecone")

# ------------------ QUERY ------------------
st.subheader("Ask a Question")
question = st.text_input("Your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state.has_data:
        st.warning("Please ingest a document first.")
        st.stop()

    retrieved_docs = st.session_state.vectorstore.similarity_search(
        question,
        k=8,
    )

    docs = rerank_docs(question, retrieved_docs, top_n=3)

    if not docs:
        st.warning("No relevant context found.")
        st.stop()

    context = "\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    try:
        response = llm.invoke(prompt)

        st.markdown("### Answer")
        st.write(response.content)

        st.markdown("### Sources")
        for i, doc in enumerate(docs):
            meta = doc.metadata
            st.markdown(
                f"[{i+1}] Source: {meta['source']} | Chunk ID: {meta['chunk_id']}"
            )

    except Exception as e:
        st.error("LLM failed.")
        st.exception(e)
