import os
import streamlit as st
import cohere
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ PAGE ------------------
st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

# ------------------ ENV ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not all([GROQ_API_KEY, COHERE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME]):
    st.error("Missing environment variables")
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

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ------------------ SESSION ------------------
if "has_data" not in st.session_state:
    st.session_state.has_data = False

# ------------------ RERANK ------------------
def rerank_docs(query, docs, top_n=3):
    texts = [d.page_content for d in docs]
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    docs = splitter.create_documents([text])

    vectors = embeddings.embed_documents([d.page_content for d in docs])

    upserts = []
    for i, vec in enumerate(vectors):
        upserts.append({
            "id": f"chunk-{i}",
            "values": vec,
            "metadata": {
                "text": docs[i].page_content,
                "chunk_id": i,
                "source": "user_input",
            },
        })

    index.upsert(vectors=upserts)
    st.session_state.has_data = True

    st.success(f"Ingested {len(upserts)} chunks into Pinecone")

# ------------------ QUERY ------------------
st.subheader("Ask a Question")
question = st.text_input("Your question")

if st.button("Ask"):
    if not st.session_state.has_data:
        st.warning("Please ingest a document first.")
        st.stop()

    q_vec = embeddings.embed_query(question)

    res = index.query(
        vector=q_vec,
        top_k=8,
        include_metadata=True,
    )

    docs = [
        Document(
            page_content=m["metadata"]["text"],
            metadata=m["metadata"],
        )
        for m in res["matches"]
    ]

    docs = rerank_docs(question, docs, top_n=3)

    context = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    st.markdown("### Answer")
    st.write(response.content)

    st.markdown("### Sources")
    for i, d in enumerate(docs):
        st.markdown(f"[{i+1}] Chunk ID: {d.metadata['chunk_id']}")
