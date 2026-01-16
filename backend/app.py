import os
import streamlit as st
import cohere
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ------------------ PAGE ------------------
st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

st.write("GROQ_API_KEY present:", bool(os.getenv("GROQ_API_KEY")))
st.write("QDRANT_URL present:", bool(os.getenv("QDRANT_URL")))

# ------------------ KEYS ------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GROQ_API_KEY or not COHERE_API_KEY or not QDRANT_URL or not QDRANT_API_KEY:
    st.error("One or more required API keys are missing.")
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

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
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
text = st.text_area("Paste text to ingest")
st.write("Text length:", len(text))

COLLECTION_NAME = "mini_rag_docs_final_v3"

if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    docs = splitter.create_documents([text])

    texts = [doc.page_content for doc in docs]

    # 🔹 Compute embeddings explicitly
    vectors = embeddings.embed_documents(texts)
    vector_size = len(vectors[0])

    # 🔹 Create collection ONCE with exact dimension
    try:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
    except Exception:
        # Collection already exists → OK
        pass

    # 🔹 Build points manually (no LangChain magic)
    points = []
    for i, vector in enumerate(vectors):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "text": texts[i],
                "source": "user_input",
                "chunk_id": i,
            },
        })


    # 🔹 Raw upsert (this NEVER fails if schema is correct)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    # 🔹 Attach LangChain vectorstore ONLY for querying
    st.session_state.vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

    st.session_state.has_data = True
    st.success(f"Ingested {len(points)} chunks into hosted vector DB")



    
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
