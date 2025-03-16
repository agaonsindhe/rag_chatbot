import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension):
        """Initialize FAISS with L2 distance and ensure float32 compatibility."""
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        """Add vector embeddings to the FAISS index."""
        embeddings = np.asarray(embeddings, dtype=np.float32)  # Ensure FAISS-compatible format
        self.index.add(embeddings)
        print(f"✅ FAISS Index Size: {self.index.ntotal} vectors stored")

    def save_index(self, file_name):
        """Save FAISS index to a file."""
        faiss.write_index(self.index, file_name)

    def load_index(self, file_name):
        """Load FAISS index from a file."""
        self.index = faiss.read_index(file_name)

    def search(self, query_embedding, top_k=3):
        """Ensure FAISS search always returns (chunk, confidence) tuples."""
        distances, indices = self.index.search(query_embedding, top_k)

        if not indices.any():
            print("No indices found")
            return []  # Ensure an empty list instead of None or string
        print("distances:", distances)
        print("indices:", indices)

        confidence_scores = 1 - (distances / np.max(distances, axis=1, keepdims=True))
        # Return a list of (index, confidence_score) for each query
        results = [
            [(idx, float(conf)) for idx, conf in zip(indices[i], confidence_scores[i])]
            for i in range(len(indices))
        ]
        return results  # Returns a list of lists (one for each query)

