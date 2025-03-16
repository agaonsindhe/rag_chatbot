from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=16):
        """Initialize embedding model with batch processing."""
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size  # Enable batch processing

    def encode(self, texts):
        """Encodes financial texts into vector embeddings, ensuring compatibility."""
        if isinstance(texts, str):
            texts = [texts]  # Convert single text to list

        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=self.batch_size)
        print("Embeddings shape: ", embeddings.shape)

        return np.asarray(embeddings, dtype=np.float32)  # Ensuring consistent format
