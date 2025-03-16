from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import yaml
import io
from contextlib import redirect_stdout
import re
from transformers import pipeline

from download_model import ensure_huggingface_model


class RAGChatbot:
    def __init__(self, model_name=None):
        """
        Initialize a Hugging Face model dynamically based on the selected model.
        """
        # Load available models from config.yaml
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Load models from config
        self.models = config["slm_models"]

        # If no model is provided, use the default one
        self.selected_model = model_name if model_name else config["default_model"]
        model_path = self.models[self.selected_model]

        print(f"Loading model: {self.selected_model} ({model_path})...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True  # Optimized memory usage
        ).to(self.device)

        print(f"{self.selected_model} successfully loaded!")

    def get_response(self, context, query):
        """
        Generate a response using structured financial data.
        """
        input_text = (
            f"Use ONLY the structured financial data below to answer the question.\n\n"
            f"DATASET:\n{context}\n\n"
            f"Follow these rules:\n"
            f"- If a specific year is mentioned, return data ONLY for that year.\n"
            f"- If a numerical threshold is mentioned, apply it to the relevant column.\n"
            f"- If the requested data is unavailable, respond with 'I don't know'.\n\n"
            f"Question: {query}\nAnswer:"
        )

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9,
                                         repetition_penalty=1.1)

        clean_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print("output",clean_output)
        # Find the starting index for "Question:"
        start_index = clean_output.find("Question:")

        # Extract the substring starting from "Question:"
        if start_index != -1:
            extracted_string = clean_output[start_index:]
            print("Extracted string:\n", extracted_string)
        else:
            extracted_string = clean_output
            print("The substring 'Question:' was not found in the text.")

        print("Extracted just before return ",extracted_string)
        return extracted_string

    # def get_response(self, context, query):
    #     """
    #     Generate a response using the dynamically selected model.
    #     """
    #     input_text = (
    #         f"Use ONLY the following information to answer:\n{context}\n"
    #         f"If the answer is not found, respond with 'I don't know'.\n"
    #         f"Question: {query}\nAnswer:\n"
    #     )
    #
    #     # Use Hugging Face pipeline for better response extraction
    #     generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer,
    #                          device=0)  # Use GPU if available
    #
    #     print("🚀 Generating response...")
    #     response_text = generator(input_text, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)[0][
    #         "generated_text"]
    #
    #     print(f"📝 Raw Model Response: {response_text}")  # Debug log
    #
    #     # Extract only the answer portion
    #     if "Answer:" in response_text:
    #         cleaned_response = response_text.split("Answer:")[-1].strip()
    #     else:
    #         cleaned_response = response_text  # Fallback
    #
    #     print(f"✅ Final Cleaned Response: {cleaned_response}")  # Debug log
    #
    #     return cleaned_response  # Return only the cleaned answer
