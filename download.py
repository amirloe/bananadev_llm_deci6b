# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    T5Tokenizer.from_pretrained("google/flan-t5-base")
    T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    SentenceTransformer('BAAI/bge-small-en')


if __name__ == "__main__":
    download_model()