import argparse
import yaml
import os
import pickle
import time

# Import libraries for UI
import streamlit as st
import gradio as gr

# Import custom modules
from src.data_processor import FinancialDataProcessor
from src.embedding_model import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.chunk_merger import ChunkMerger
from src.rag_chatbot import RAGChatbot
from src.guardrails import Guardrails

# Parse command-line arguments for mode selection
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["gradio", "streamlit"], default="gradio", help="Choose the UI mode")
args = parser.parse_args()

# Load Configurations
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Initialize Components
processor = FinancialDataProcessor(config["data_path"])
chunks = processor.preprocess_data()
embedding_model = EmbeddingModel(config["embedding_model"])
embedding_cache_path = config["embedding_cache_path"]

# Load or compute embeddings
if os.path.exists(embedding_cache_path):
    print("✅ Loading cached embeddings...")
    with open(embedding_cache_path, "rb") as f:
        chunk_embeddings = pickle.load(f)
else:
    print("Computing embeddings (first time only)...")
    chunk_embeddings = embedding_model.encode(chunks)
    with open(embedding_cache_path, "wb") as f:
        pickle.dump(chunk_embeddings, f)

vector_store = FAISSVectorStore(chunk_embeddings.shape[1])
vector_store.add_embeddings(chunk_embeddings)
vector_store.save_index(config["index_path"])
vector_store.load_index(config["index_path"])

chunk_merger = ChunkMerger(chunks, processor.data, embedding_model, vector_store)
guardrails = Guardrails()

# Function to load chatbot
def load_chatbot(model_choice):
    return RAGChatbot(model_name=model_choice)

# Function to process queries
def get_chatbot_response(query, model_choice, rag_mode):
    chatbot = load_chatbot(model_choice)
    response = "⚠️ No relevant information found in the dataset."

    if guardrails.validate_input(query):
        retrieved_chunks = chunk_merger.retrieve_chunks(query, top_k=config["retrieval"]["top_k"])
        print(f"🔍 FAISS Retrieved Chunks (Count: {len(retrieved_chunks)}):", retrieved_chunks)

        if rag_mode == "Basic RAG":
            merged_text = "\n".join(retrieved_chunks)
        else:
            merged_text = chunk_merger.adaptive_merge_chunks(query, retrieved_chunks)

        if retrieved_chunks:
            response = chatbot.get_response(merged_text, query)
            response = guardrails.filter_output(response)

    return response

# ----------------------------------------------
# 🚀 **Gradio UI**
# ----------------------------------------------
if args.mode == "gradio":
    print("🟢 Running Gradio UI...")

    def chatbot_interface(query, model_choice, rag_mode):
        start_time = time.time()
        response = get_chatbot_response(query, model_choice, rag_mode)
        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        return response, f"⏳ Response Time: {response_time} seconds"

    # Define Gradio Interface
    interface = gr.Interface(
        fn=chatbot_interface,
        inputs=[
            gr.Textbox(label="Enter your financial question"),
            gr.Radio(list(config["slm_models"].keys()), label="Choose a model", value=config["default_model"]),
            gr.Radio(["Basic RAG", "Advanced RAG"], label="Choose Retrieval Mode", value="Basic RAG"),
        ],
        outputs=[
            gr.Textbox(label="Response"),
            gr.Textbox(label="Processing Time"),
        ],
        title="💬 Financial RAG Chatbot",
        description="A chatbot powered by Retrieval Augmented Generation (RAG) for financial queries.",
    )

    interface.launch(share=True)  # Launch Gradio App

# ----------------------------------------------
# 🟢 **Streamlit UI**
# ----------------------------------------------
elif args.mode == "streamlit":
    print("🟢 Running Streamlit UI...")

    st.sidebar.title("⚙️ Model Selection")
    model_choice = st.sidebar.selectbox("Choose a model:", list(config["slm_models"].keys()), index=list(config["slm_models"].keys()).index(config["default_model"]))
    rag_mode = st.sidebar.radio("Choose Retrieval Mode:", ["Basic RAG", "Advanced RAG"])

    # Preview dataset button
    if st.sidebar.button("Preview Dataset"):
        st.write("### Financial Dataset Preview")
        st.dataframe(processor.data.head(10))

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Layout with two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h2 style='text-align: center;'>💬 Financial RAG Chatbot</h2>", unsafe_allow_html=True)
        query = st.text_input("Enter your financial question:")

        if st.button("Get Answer"):
            start_time = time.time()
            response = get_chatbot_response(query, model_choice, rag_mode)
            end_time = time.time()
            response_time = round(end_time - start_time, 2)

            if response:
                st.write("### Response:")
                st.markdown(f"```\n{response}\n```")
            else:
                st.warning("⚠️ No response generated.")

            st.sidebar.write(f"⏳ Response Time: {response_time} seconds")
            st.session_state.chat_history.append((query, response, rag_mode))

    # Scrollable chat history
    with col2:
        st.markdown("<h2 style='text-align: center;'>📜 Chat History</h2>", unsafe_allow_html=True)

        for q, r, mode in reversed(st.session_state.chat_history):
            rag_color = "#ff5733" if mode == "Basic RAG" else "#33b5e5"
            st.markdown(f"<p style='color:{rag_color}; font-weight: bold;'>{mode}</p>", unsafe_allow_html=True)
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {r}")
            st.write("---")

