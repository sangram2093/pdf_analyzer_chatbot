from flask import Flask, render_template, request, jsonify
import os
import vertexai
import google.auth
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

app = Flask(__name__)

# Set environment
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "updated_keyfile_gcp.json"
credentials = google.auth.default()
vertexai.init(project="project_name", location="europe-west3", credentials=credentials)

# Initialize LLM and Embeddings
llm = VertexAI(model_name="gemini-1.5-pro-002", max_output_tokens=4096)
embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")

# Load and split PDF
loader = PyPDFLoader("Policy.pdf", extract_images=False)
documents = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20, length_function=len)
chunks = text_splitter.split_documents(documents)

# Vector DB setup
persist_directory = "test_database"
vectordb = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory=persist_directory)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    result = qa_chain.invoke({"query": question})
    return jsonify({"answer": result["result"]})

if __name__ == "__main__":
    app.run(debug=True)
