from typing import Optional
import uuid
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Request(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stop: str
    penalty_alpha: Optional[float] = None
    repetition_penalty: float = 1.0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Fast API for huggingface CausalLM models")
    parser.add_argument("model_name")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name,
                                                 torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if torch.cuda.is_available():
        model.to("cuda:0")

    app = FastAPI()

    @app.get("/alive")
    def health():
        return {"status": "alive"}

    @app.post("/completions")
    def generate(request: Request):
        encoded_input = tokenizer(request.prompt, return_tensors='pt')
        result = model.generate(
            input_ids=encoded_input['input_ids'].cuda(0) if torch.cuda.is_available() else encoded_input['input_ids'],
            do_sample=True,
            max_new_tokens=request.max_tokens,
            num_return_sequences=1,
            top_p=request.top_p,
            temperature=request.temperature,
            penalty_alpha=request.penalty_alpha,
            top_k=request.top_k,
            output_scores=False,
            return_dict_in_generate=False,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=tokenizer.convert_tokens_to_ids(request.stop),
            use_cache=True
        )
        result = tokenizer.decode(result[0], skip_special_tokens=True)
        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "model": args.model_name,
            "choices": [
                {
                    "text": result,
                    "index": 0
                }
            ]
        }

    uvicorn.run(app, port=args.port, workers=1)
