# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    checkpoint = "Deci/DeciLM-6b-instruct"
    device = "cuda"  if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)


if __name__ == "__main__":
    download_model()