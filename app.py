import numpy as np
from potassium import Potassium, Request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import torch, gc

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    checkpoint = "Deci/DeciLM-6b-instruct"
    device = "cuda"  if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    context = {
        "model": model,
        "tokenizer": tokenizer,
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    """
    test
    """
    device = "cuda"  if torch.cuda.is_available() else "cpu"
    gc.collect()
    torch.cuda.empty_cache()
    prompts = request.json.get("prompts")
    output_tokens = []
    for text in prompts:
        inputs = context['tokenizer'].encode(text, return_tensors="pt").to(device)
        outputs = context['model'].generate(inputs, max_new_tokens=20, do_sample=False, top_p=0.95)
        out_text = context['tokenizer'].decode(outputs[0])
        # find with regex the token after  #Answer: and before <\s>
        # output_token = re.search(r"#Answer: (.*)<\s>", out_text).group(1)
        output_token = re.findall(r"#Answer:\n(.*)</s>", out_text)
        output_token = output_token[0] if len(output_token) > 0 else ""
        output_tokens.append(output_token)


    return Response(
        json={
            "outputs": output_tokens},
        status=200
    )

    


if __name__ == "__main__":
    #test a
    app.serve()
