import os
import json
import pandas as pd
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataPreprocessor:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        """
        Initializes the preprocessor with chunking parameters.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text.
        """
        text = ""
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()

    def extract_text_from_json(self, json_path):
        """
        Extracts text from a JSON file containing financial statements.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            str: Extracted text.
        """
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return "\n".join([f"{key}: {value}" for key, value in data.items()])

    def extract_text_from_csv(self, csv_path):
        """
        Extracts text from a CSV file by converting it into a readable format.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            str: Extracted text.
        """
        df = pd.read_csv(csv_path)
        return df.to_string(index=False)

    def chunk_text(self, text):
        """
        Splits text into smaller chunks for embedding.

        Args:
            text (str): The text to be chunked.

        Returns:
            list: List of text chunks.
        """
        return self.text_splitter.split_text(text)

    def process_file(self, file_path):
        """
        Processes a file (PDF, JSON, CSV) and returns its text chunks.

        Args:
            file_path (str): Path to the file.

        Returns:
            list: List of text chunks.
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif file_extension == ".json":
            text = self.extract_text_from_json(file_path)
        elif file_extension == ".csv":
            text = self.extract_text_from_csv(file_path)
        else:
            raise ValueError("Unsupported file format! Only PDF, JSON, and CSV are supported.")

        return self.chunk_text(text)

# Example Usage:
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Example paths (Update with actual file paths)
    file_path = "data/sample_financials.pdf"
    
    chunks = preprocessor.process_file(file_path)
    
    print(f"Extracted {len(chunks)} chunks from {file_path}")
    for i, chunk in enumerate(chunks[:5]):  # Display first 5 chunks
        print(f"Chunk {i+1}: {chunk}\n")
