### Financial RAG Chatbot

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** to answer financial questions based on financial statements from the last two years. The chatbot is built using **FAISS for vector search**, **SBERT for embedding generation**, and **GPT4All for response generation**. It also features a **Streamlit UI** for user interaction.

## 🚀 Features
- **Basic RAG**: Embeds and retrieves financial data using FAISS.
- **Advanced RAG**: Implements **chunk merging & adaptive retrieval**.
- **Interactive UI**: Built using **Streamlit**.
- **Modular Structure**: Follows **SOLID principles** for maintainability.

## 📁 Project Structure
```
rag_chatbot/
│── data/
│   ├── financial_statements.csv
│── src/
│   ├── __init__.py
│   ├── data_processor.py         # Handles data loading & chunking
│   ├── embedding_model.py        # Handles text embeddings
│   ├── vector_store.py           # Handles FAISS storage & retrieval
│   ├── chunk_merger.py           # Implements chunk merging & adaptive retrieval
│   ├── rag_chatbot.py            # RAG chatbot for response generation
│── main.py                       # Streamlit UI & chatbot execution
│── requirements.txt               # Dependencies
│── README.md                      # Documentation
```

## 🛠️ Installation
1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
```

2️⃣ **Create a virtual environment** (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Running the Chatbot
```bash
streamlit run main.py
```

## 📦 Dependencies
All dependencies are listed in **`requirements.txt`**.

## 📌 Future Enhancements
- ✅ Add **confidence scores** to responses.
- ✅ Implement **response filtering** for guardrails.
- ✅ Enhance **adaptive retrieval** strategies.

## 📄 License
This project is licensed under **BITS License**.