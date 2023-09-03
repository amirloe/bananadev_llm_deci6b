import numpy as np
from potassium import Potassium, Request, Response
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import torch, gc

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")
    # sentence_model = SentenceTransformer('BAAI/bge-small-en')
    context = {
        "model": model,
        "tokenizer": tokenizer,
        "sentence_model": None,
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    """
    test
    """
    
    gc.collect()
    torch.cuda.empty_cache()
    if request.json.get("type", "") == "yesno":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prompts = request.json.get("prompts")
        word_list = request.json.get("word_list")
        model = context.get("model")
        tokenizer = context.get("tokenizer")
        tokens = [tokenizer.encode(word)[0] for word in word_list]
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids.to(device)
        decoder_input_ids = tokenizer(prompts, return_tensors="pt", padding=True).input_ids
        decoder_input_ids = model._shift_right(decoder_input_ids).to(device)
        output = model(input_ids, decoder_input_ids=decoder_input_ids)
        logits = output.logits.to("cpu").detach().numpy()
        words_logits = logits[:, 0, tokens]
        output = np.argmax(words_logits, axis=1)

        return Response(
            json={"logits": words_logits.tolist(),
                "output": output.tolist()},
            status=200
        )
    elif request.json.get("type", "") == "sentence":
        model = context.get("sentence_model")
        sentence = request.json.get("sentence")
        embedding = model.encode([sentence])[0]
        return Response(
            json={"embedding": embedding.tolist()},
            status=200
        )
    


if __name__ == "__main__":
    #test
    app.serve()
