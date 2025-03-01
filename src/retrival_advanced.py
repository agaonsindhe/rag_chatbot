import numpy as np
from rank_bm25 import BM25Okapi
from embeddings import EmbeddingModel, VectorDatabase
from preprocessing import DataPreprocessor

class AdvancedRetriever:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2", index_path="models/vector_index.pkl"):
        """
        Initializes the Advanced RAG retriever using Hybrid Retrieval (BM25 + FAISS).

        Args:
            embedding_model (str): Pre-trained embedding model.
            index_path (str): Path to FAISS vector index.
        """
        self.embedder = EmbeddingModel(model_name=embedding_model)
        self.vector_db = VectorDatabase(index_path=index_path)
        self.bm25 = None
        self.text_chunks = []

    def build_bm25(self, texts):
        """
        Builds the BM25 index for keyword-based search.

        Args:
            texts (list): List of text chunks.
        """
        tokenized_texts = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        self.text_chunks = texts

    def hybrid_retrieve(self, query, top_k=5):
        """
        Retrieves relevant text chunks using Hybrid Retrieval (BM25 + FAISS).

        Args:
            query (str): The user query.
            top_k (int): Number of top results to retrieve.

        Returns:
            list: Top matching text chunks.
        """
        # Step 1: BM25 Keyword Search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_results = [self.text_chunks[i] for i in bm25_top_indices]

        # Step 2: FAISS Embedding Search
        query_embedding = self.embedder.generate_embeddings([query]).reshape(1, -1)
        faiss_results = self.vector_db.search(query_embedding, top_k=top_k)

        # Step 3: Merge & Re-Rank Results
        combined_results = list(set(bm25_results + faiss_results))  # Remove duplicates
        return combined_results[:top_k]

# Example Usage:
if __name__ == "__main__":
    retriever = AdvancedRetriever()

    # Load and process sample text chunks
    sample_texts = [
        "The company's revenue increased by 12% in Q3.",
        "Net profit declined due to increased operational costs.",
        "Stock prices surged after the earnings report.",
        "Revenue projections for the next quarter remain strong."
    ]
    
    retriever.build_bm25(sample_texts)  # Build BM25 index

    query = "How did the company's revenue change?"
    results = retriever.hybrid_retrieve(query, top_k=3)

    print("Top Matching Results:")
    for result in results:
        print(f"- {result}")
