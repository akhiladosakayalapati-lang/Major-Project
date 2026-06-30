# Major-Project
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="QuiryDocAI", layout="wide")

import base64

# Convert image to base64
with open("logo.png", "rb") as f:
    data = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
.header {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.header img {{
    width: 55px;
}}
.header-text h2 {{
    margin: 0;
}}
.header-text p {{
    margin: 0;
    color: gray;
}}
</style>

<div class="header">
    <img src="data:image/png;base64,{data}">
    <div class="header-text">
        <h2>QuiryDocAI</h2>
        <p>Chat with your PDF</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="DocuTrustAI",
    page_icon="logo.png",
    layout="wide"
)
# ---------------- MODEL ----------------
if "model" not in st.session_state:
    with st.spinner("Loading AI model... ⏳"):
        st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    st.success("Model loaded ✅")

model = st.session_state.model

# ---------------- FUNCTIONS ----------------
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def split_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

def create_index(chunks):
    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index

def search(query, index, chunks):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    return [chunks[i] for i in I[0]]

def generate_answer(query, results):
    # Combine top chunks into a better answer
    context = " ".join(results)
    answer = f"👉 Based on the document:\n\n{context}"
    return answer

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and not st.session_state.processed:

    with st.spinner("Reading PDF..."):
        text = extract_text(uploaded_file)

    if text.strip() == "":
        st.error("❌ No text found in PDF")
    else:
        with st.spinner("Processing document..."):
            chunks = split_text(text)
            index = create_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.processed = True
        st.success("✅ PDF ready! Ask your questions below 👇")

# ---------------- CHAT INTERFACE ----------------
if st.session_state.processed:

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # User input
    query = st.chat_input("Ask something about your PDF...")

    if query:
        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        # Generate answer
        with st.spinner("Thinking... 🤔"):
            results = search(query, st.session_state.index, st.session_state.chunks)
            answer = generate_answer(query, results)

        # Save assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)
st.markdown("---")
