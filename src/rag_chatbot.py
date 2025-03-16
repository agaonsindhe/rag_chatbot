from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml
from download_model import ensure_huggingface_model

class RAGChatbot:
    def __init__(self, model_name=None):
        """
        Initialize a Hugging Face model dynamically based on the selected model.
        """
        # Load available models from config.yaml
        with open("config.yaml", "r") as file:
            config = yaml.safe_load(file)

        # Resolve model name safely
        self.selected_model = model_name if model_name else config.get("default_model", None)
        if not self.selected_model:
            raise ValueError("Error: No valid model found in config.yaml")

        model_path = config["model_path"] + self.selected_model
        ensure_huggingface_model(self.selected_model, model_path)

        print(f"🔹 Loading model: {self.selected_model} ({model_path})...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model with optimized settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True  # Optimized memory usage
        ).to(self.device)

        print(f"✅ {self.selected_model} successfully loaded!")

    def get_response(self, context, query):
        """
        Generate a response using structured financial data with improved formatting.
        """
        # **Enhancement: Format context as structured financial data**
        formatted_context = "\n".join(
            f"- {line}" for line in context.split("\n") if line.strip()
        )

        input_text = (
            f"Use ONLY the structured financial data below to answer the question.\n\n"
            f"DATASET:\n{context}\n\n"
            f"Follow these rules:\n"
            f"- If a specific year is mentioned, return data ONLY for that year.\n"
            f"- If a numerical threshold is mentioned, apply it to the relevant column.\n"
            f"- If the requested data is unavailable, respond with 'I don't know'.\n\n"
            f"Question: {query}\n\nAnswer:"
        )

        print("input prompt: ",input_text)
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, repetition_penalty=1.1)

        clean_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print("response: ",clean_output)
        # **Enhancement: Clean up unnecessary preamble in the response**
        answer_start = clean_output.find("Answer:")
        print("answer start: ",answer_start)
        if answer_start != -1:
            print("inside if: ",answer_start)
            extracted_answer = clean_output[answer_start + len("Answer:") :].strip()
        else:
            print("inside else: ", clean_output.strip())
            extracted_answer = clean_output.strip()

        print("✅ Extracted Answer:\n", extracted_answer)
        return extracted_answer
