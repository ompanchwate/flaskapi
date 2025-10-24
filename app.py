from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ✅ Load Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in environment")

# ✅ Load and chunk PDF
def load_pdf_chunks(pdf_path, max_words=200):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += ' '.join([block[4] for block in page.get_text("blocks")])
    paragraphs = full_text.split("\n\n")
    chunks = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_words:
            chunks.append(para)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i+max_words])
                chunks.append(chunk)
    return chunks

# ✅ Document class
class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id or str(hash(page_content))

# ✅ Initialize vectorstore
pdf_path = "DevSecOps.pdf"  # Make sure this file is present in your Render repo
chunks = load_pdf_chunks(pdf_path)
documents = [Document(page_content=c) for c in chunks]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

# ✅ Initialize Groq LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7, groq_api_key=groq_api_key)

# ✅ Flask app
app = Flask(__name__)
chat_history = []

@app.route("home", methods=["GET"])
def home():
    print("Hello there!")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "Missing 'question' in request"}), 400

    # 🔍 Retrieve context
    docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 🧠 Build prompt with memory
    history_text = ""
    for i, (q, a) in enumerate(chat_history):
        history_text += f"Previous Q{i+1}: {q}\nPrevious A{i+1}: {a}\n"

    prompt = f"""You are a helpful DevSecOps assistant. Use the context and chat history below to answer the question.

Context:
{context}

Chat History:
{history_text}

Current Question:
{query}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content
    chat_history.append((query, answer))

    return jsonify({"answer": answer})
