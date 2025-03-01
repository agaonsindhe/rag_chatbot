import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.

        Args:
            model_name (str): Name of the Sentence Transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of text chunks.

        Args:
            texts (list): List of text chunks.

        Returns:
            np.array: Numpy array of embeddings.
        """
        return np.array(self.model.encode(texts, convert_to_numpy=True))

class VectorDatabase:
    def __init__(self, embedding_dim=384, index_path="models/vector_index.pkl"):
        """
        Initializes the FAISS vector database.

        Args:
            embedding_dim (int): Dimension of embeddings.
            index_path (str): Path to save the FAISS index.
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.text_chunks = []

        # Load existing index if available
        if os.path.exists(self.index_path):
            self.load_index()

    def add_embeddings(self, texts, embeddings):
        """
        Adds new text chunks and their embeddings to the FAISS index.

        Args:
            texts (list): List of text chunks.
            embeddings (np.array): Corresponding embeddings.
        """
        self.index.add(embeddings)
        self.text_chunks.extend(texts)
        self.save_index()

    def search(self, query_embedding, top_k=5):
        """
        Searches for the top_k most relevant text chunks.

        Args:
            query_embedding (np.array): Query embedding.
            top_k (int): Number of results to return.

        Returns:
            list: Top matching text chunks.
        """
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]

    def save_index(self):
        """ Saves the FAISS index and text chunks. """
        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self.index, "texts": self.text_chunks}, f)

    def load_index(self):
        """ Loads the FAISS index and text chunks. """
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.text_chunks = data["texts"]

# Example Usage:
if __name__ == "__main__":
    texts = ["Revenue increased by 10%", "The company reported a loss", "Net profit rose in Q3"]
    
    # Initialize embedding model
    embedder = EmbeddingModel()
    embeddings = embedder.generate_embeddings(texts)

    # Initialize vector database
    vector_db = VectorDatabase()
    vector_db.add_embeddings(texts, embeddings)

    # Query the database
    query = ["What was the company's revenue growth?"]
    query_embedding = embedder.generate_embeddings(query).reshape(1, -1)
    results = vector_db.search(query_embedding)

    print("Top Matching Results:")
    for result in results:
        print(f"- {result}")
