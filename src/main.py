import streamlit as st
import yaml
import os
import pickle
import time

from data_processor import FinancialDataProcessor
from embedding_model import EmbeddingModel
from vector_store import FAISSVectorStore
from chunk_merger import ChunkMerger
from rag_chatbot import RAGChatbot
from guardrails import Guardrails
import os
import pickle
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
import sys

print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
@st.cache_resource
def load_chatbot(model_choice):
    return RAGChatbot(model_name=model_choice)

# Load Configurations
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config()

# Streamlit UI for model selection
st.sidebar.title("⚙️ Model Selection")
# Get list of model keys
model_keys = list(config["slm_models"].keys())

# Ensure default model key exists
default_model_key = config["default_model"]
if default_model_key not in model_keys:
    raise ValueError(f"Error: Default model '{default_model_key}' not found in slm_models!")

# Select a model using keys
selected_model_key = st.sidebar.selectbox(
    "Choose a model:",
    model_keys,
    index=model_keys.index(default_model_key)
)

# Resolve selected model name
model_choice = config["slm_models"][selected_model_key]  # Convert key to full model path
print("Model choice",model_choice)

# Streamlit UI for selecting RAG mode
rag_mode = st.sidebar.radio("Choose Retrieval Mode:", ["Basic RAG", "Advanced RAG"])

# Initialize Components
processor = FinancialDataProcessor(config["data_path"])
chunks = processor.preprocess_data()
# print("chunks from loaded data ",chunks)
embedding_model = EmbeddingModel(config["embedding_model"])
embedding_cache_path = config["embedding_cache_path"]
# Add dataset preview option
if st.sidebar.button("Preview Dataset"):
    st.write("### Financial Dataset Preview")
    st.dataframe(processor.data.head(10))

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
# Ensure embeddings are in float32 format (FAISS requirement)
chunk_embeddings = chunk_embeddings.astype("float32")
vector_store.add_embeddings(chunk_embeddings)
vector_store.save_index(config["index_path"])
vector_store.load_index(config["index_path"])

# Initialize Retrieval, Chatbot & Guardrails
chunk_merger = ChunkMerger(chunks, processor.data,embedding_model, vector_store)
chatbot = load_chatbot(model_choice)
guardrails = Guardrails()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Layout with two columns and a vertical divider
col1, col2 = st.columns([2, 1])  # Middle column for spacing

with col1:
    st.markdown("<h2 style='text-align: center;'>💬 Financial RAG Chatbot</h2>", unsafe_allow_html=True)
    query = st.text_input("Enter your financial question:")
    if st.button("Get Answer"):
        start_time = time.time()  # Track response time
        response = "⚠No relevant information found in the dataset."
        if guardrails.validate_input(query):  # Validate user input
            retrieved_chunks = chunk_merger.retrieve_chunks(query, top_k=config["retrieval"]["top_k"])
            print(f"FAISS Retrieved Chunks (Before Processing): {retrieved_chunks}")

            if isinstance(retrieved_chunks, str) or not retrieved_chunks:
                print("FAISS did not return any relevant results. Using fallback.")
                retrieved_chunks = []
                retrieved_texts = []
                confidence_scores = []
            else:
                if isinstance(retrieved_chunks[0], tuple) and len(retrieved_chunks[0]) == 2:
                    retrieved_texts = [chunk for chunk, confidence in retrieved_chunks]
                    confidence_scores = [confidence for _, confidence in retrieved_chunks]
                else:
                    print("FAISS returned only text chunks without confidence scores. Fixing format...")
                    retrieved_texts = retrieved_chunks
                    confidence_scores = ["N/A"] * len(retrieved_chunks)

            if not retrieved_texts:
                response = "No relevant financial data found in the dataset."
                confidence_scores = []
            else:
                response = chatbot.get_response("\n".join(retrieved_texts), query)
                response = guardrails.filter_output(response)  # Ensure output filtering

        else:
            st.warning("Query blocked: This request does not match financial topics.")

        end_time = time.time()
        response_time = round(end_time - start_time, 2)
        st.sidebar.write(f"⏳ Response Time: {response_time} seconds")

        # Store query, response, RAG mode, and confidence scores in chat history
        st.session_state.chat_history.append((query, response, rag_mode, confidence_scores))

st.divider()

# Scrollable chat history
with col2:
    st.markdown("""
        <style>
        .chat-history {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: #f9f9f9;
        }
        </style>
        <div class="chat-history">
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center;'>📜 Chat History</h2>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.write("🔍 No chat history yet. Ask a question to start the conversation!")

    else:
        with st.container():  # Make chat history scrollable dynamically
            for q, r, mode, confidence_scores in reversed(st.session_state.chat_history):
                rag_color = "#ff5733" if mode == "Basic RAG" else "#33b5e5"  # Color for each mode
                # Ensure confidence_scores are floats before rounding
                confidence_display = ", ".join(
                    [f"{round(float(score) * 100, 2)}%" for score in confidence_scores if
                     isinstance(score, (int, float))]
                ) or "N/A"

                st.markdown(
                    f"""
                    <p style='color:{rag_color}; font-weight: bold;'>{mode}</p>
                    <p><b>Q:</b> {q}</p>
                    <p><b>A:</b> {r}</p>
                    <p>🔍 <b>Confidence Scores:</b> {confidence_display}</p>
                    <hr style="border:1px solid #ddd;">
                    """,
                    unsafe_allow_html=True
                )

    st.markdown("</div>", unsafe_allow_html=True)  # Close scrollable div

