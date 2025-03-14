from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import yaml
import io
from contextlib import redirect_stdout
import re

from download_model import ensure_huggingface_model


class RAGChatbot:
    def __init__(self, model_name=None):
        """
        Initialize a Hugging Face model dynamically based on the selected model.
        """
        # Load available models from config.yaml
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        self.models = config["slm_models"]

        # If no model is provided, use the default one
        self.selected_model = model_name if model_name else config["default_model"]
        model_path = self.models[self.selected_model]

        ensure_huggingface_model(self.selected_model, model_path)

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
        Generate a response using the dynamically selected model.
        """
        input_text = (
            f"Use ONLY the following information to answer:\n{context}\n"
            f"If the answer is not found, respond with 'I don't know'.\n"
            f"Question: {query}\nAnswer:\n"
        )

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                return_dict_in_generate=True,  # Ensure structured return
                output_scores=True  # Track generated tokens
            )

        # Decode the model's output to text
        response_text = self.tokenizer.decode(output_ids["sequences"][0], skip_special_tokens=True)

        print(f"📝 Raw Model Response: {response_text}")  # Debug log

        # Extract only the answer portion
        if "Answer:" in response_text:
            cleaned_response = response_text.split("Answer:")[-1].strip()
        else:
            cleaned_response = response_text  # Fallback

        print(f"✅ Final Cleaned Response: {cleaned_response}")  # Debug log

        return cleaned_response  # Return only the cleaned answer
