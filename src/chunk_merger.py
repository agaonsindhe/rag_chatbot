from sentence_transformers import SentenceTransformer
import numpy as np
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ChunkMerger:
    def __init__(self, chunks, df, embedding_model, vector_store):
        """
        Initialize ChunkMerger with dataset, embedding model, and FAISS vector store.
        """
        self.df = df  # Store dataset DataFrame
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight model for similarity scoring

    def retrieve_chunks(self, query, top_k=3):
        """
        Retrieve top-k relevant chunks based on query embedding and apply structured filtering.
        """
        print(f"New Query: {query}")
        query_embedding = self.embedding_model.encode([query])

        # **Updated to handle confidence scores**
        search_results = self.vector_store.search(query_embedding, top_k)
        retrieved_chunks = [(self.chunks[i], confidence) for i, confidence in search_results[0]]

        print(f"FAISS Returned Indices & Confidence Scores: {retrieved_chunks}")

        # Convert retrieved chunks into structured DataFrame format
        column_names = list(self.df.columns)  # Dynamically extract column names
        # column_names.append("Confidence")
        data_rows = []
        confidence_scores = []
        print("column names ",column_names)
        for chunk, confidence in retrieved_chunks:
            print(f"Chunk: {chunk}")
            print(f"Confidence: {confidence}")
            fields = chunk.split(", ")  # Split each chunk properly
            if len(fields) == len(column_names):  # Ensure proper alignment
                print("fields: ",fields)
                data_rows.append(fields)
                confidence_scores.append(confidence)

        df_filtered = pd.DataFrame(data_rows, columns=column_names)
        df_filtered["Confidence Score"] = confidence_scores  # Add confidence scores to DataFrame

        # **Extract numerical constraints from the query**
        year_match = re.search(r"\b(19|20)\d{2}\b", query)  # Extract year
        income_match = re.search(r"\b\d+\b", query)  # Extract numeric threshold

        # **Apply year filtering**
        if "Year" in column_names and year_match:
            target_year = year_match.group(0)
            df_filtered = df_filtered[df_filtered["Year"] == target_year]
            print(f"📅 Year Filter Applied: {target_year}")

        # **Apply financial threshold filtering**
        if income_match:
            income_threshold = float(income_match.group(0))
            if "Revenue" in column_names:
                df_filtered = df_filtered[df_filtered["Revenue"].astype(float) > income_threshold]
            elif "Net Profit" in column_names:
                df_filtered = df_filtered[df_filtered["Net Profit"].astype(float) > income_threshold]
            print(f"💰 Income Filter Applied: > {income_threshold}")


        return df_filtered

    def merge_chunks(self, query, top_k=3):
        """Retrieve & merge top-k relevant financial chunks."""
        retrieved_chunks = self.retrieve_chunks(query, top_k)
        return "\n".join(retrieved_chunks)

    def adaptive_merge_chunks(self, query, retrieved_chunks):
        """
        Advanced RAG: Merge retrieved chunks based on semantic similarity.
        """
        chunk_embeddings = self.similarity_model.encode(retrieved_chunks)
        query_embedding = self.similarity_model.encode([query])[0]

        # **Optimized: Using sklearn for cosine similarity**
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]

        # Sort chunks by similarity
        sorted_chunks = [chunk for _, chunk in sorted(zip(similarities, retrieved_chunks), reverse=True)]

        # Merge top 2 most similar chunks
        return " ".join(sorted_chunks[:2])  # Use only top 2 most relevant chunks
