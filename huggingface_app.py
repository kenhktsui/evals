from typing import Optional, List, Literal
import uuid
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    penalty_alpha: Optional[float] = None
    repetition_penalty: float = 1.0


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    penalty_alpha: Optional[float] = None
    repetition_penalty: float = 1.0


def merge_same_role_message(messages: List[Message]) -> List[Message]:
    """Merge messages with the same role."""
    merged_messages = []
    for msg in messages:
        if merged_messages and msg.role == merged_messages[-1].role:
            merged_messages[-1].content += msg.content
        else:
            merged_messages.append(msg)
    return merged_messages


def messages_to_text_prompt(messages: List[Message],
                            user_prefix: str,
                            assistant_prefix: str,
                            eos_token: Optional[str] = "\n",
                            for_completion: bool = True,
                            treat_system_to_user: bool = True) -> str:
    """Convert a list of messages to a text prompt for a model."""
    if treat_system_to_user:
        for m in messages:
            if m.role == "system":
                m.role = "user"

        messages = merge_same_role_message(messages)

    if len(messages) == 1:
        return messages[0].content

    text = ""
    for msg in messages:
        role = msg.role
        prefix = user_prefix if role == "user" else assistant_prefix
        text += f"{prefix}{msg.content}{eos_token}"
    if for_completion:
        text += assistant_prefix
    return text.lstrip()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Fast API for huggingface CausalLM models")
    parser.add_argument("model_name")
    parser.add_argument("--user_prefix", type=str, default=5000)
    parser.add_argument("--assistant_prefix", type=str, default=5000)
    parser.add_argument("--eos_token", type=str, default=5000)
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
    def completions(request: CompletionRequest):
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
            eos_token_id=tokenizer.convert_tokens_to_ids(args.eos_token),
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


    @app.post("/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        prompt = messages_to_text_prompt(request.messages,
                                         args.user_prefix,
                                         args.assistant_prefix,
                                         args.eos_token)
        encoded_input = tokenizer(prompt, return_tensors='pt')
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
            eos_token_id=tokenizer.convert_tokens_to_ids(args.eos_token),
            use_cache=True
        )
        result = tokenizer.decode(result[0], skip_special_tokens=True)

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "model": args.model_name,
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": result
                    },
                    "index": 0
                }
            ]
        }


    uvicorn.run(app, port=args.port, workers=1)
