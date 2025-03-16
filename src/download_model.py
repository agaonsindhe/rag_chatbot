import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ensure_huggingface_model(model_name, model_path):
    """
    Ensures the Hugging Face model and tokenizer exist.
    If not found, downloads and saves them to the specified directory.
    """
    print("saving model to ",model_path)
    print("model name ",model_name)
    if not os.path.exists(model_path):
        print(f"Downloading Hugging Face model: {model_name}...")
        os.makedirs(model_path, exist_ok=True)

        # Download model & tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True  # Prevents excessive memory usage
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
      
        # Save locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

        print(" Model download complete!")
    else:
        print(" Model already exists.")


if __name__ == "__main__":
    import yaml

    # Load model configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_name = config["slm_model"]
    model_path = f"models/{model_name.replace('/', '_')}"

    # Download the model if not available
    ensure_huggingface_model(model_name, model_path)
