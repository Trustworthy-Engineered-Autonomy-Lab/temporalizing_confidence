from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time
import torch
from transformers import AutoTokenizer

# === Initialize vLLM model ===
model_id = "google/gemma-3-12b-it"
llm = LLM(model=model_id, tensor_parallel_size=torch.cuda.device_count(), disable_custom_all_reduce=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", use_fast=True)

# === Create FastAPI app ===
app = FastAPI()

# === Input/output schema ===
class QueryRequest(BaseModel):
    user_message: str

class TokenLogit(BaseModel):
    token: str
    top_logits: list  # [{token, logit}, ...]

class QueryResponse(BaseModel):
    response: str
    elapsed_time: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    logprob: float
    token_level_logits: list[TokenLogit]

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(query: QueryRequest):
    user_text = query.user_message

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

    # Tokenize with chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # vLLM sampling params
    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=0.0,  # deterministic
        logprobs=5,       # top-k logprobs per token
    )

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    elapsed_time = time.time() - start_time

    output_obj = outputs[0]
    decoded = output_obj.outputs[0].text
    output_tokens = len(output_obj.outputs[0].token_ids)
    input_tokens = len(output_obj.prompt_token_ids)
    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

    # === Token-level logits ===
    token_level_logits = []
    logprob_sum = 0.0

    for logprob_dict, tok_id in zip(output_obj.outputs[0].logprobs, output_obj.outputs[0].token_ids):
        token_str = tokenizer.decode([tok_id])
        top_logits = [
            {"token": tokenizer.decode([tid]), "logit": float(lp.logprob)}
            for tid, lp in logprob_dict.items()
        ]
        token_level_logits.append({"token": token_str, "top_logits": top_logits})

        if tok_id in logprob_dict:
            logprob_sum += logprob_dict[tok_id].logprob

    return {
        "response": decoded,
        "elapsed_time": round(elapsed_time, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "logprob": round(logprob_sum, 4),
        "token_level_logits": token_level_logits
    }

