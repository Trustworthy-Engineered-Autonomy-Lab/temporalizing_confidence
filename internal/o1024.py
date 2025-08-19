from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import time

# === Initialize the model ===
model_id = "google/gemma-3-12b-it"
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch_dtype
).eval()

print(f"Model main device: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

processor = AutoProcessor.from_pretrained(
    model_id,
    use_fast=True  # 👈 启用 fast processor
)
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
    token_level_logits: list[TokenLogit]  # 每个输出 token 的 logit 概览

# === Route ===
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(query: QueryRequest):
    user_text = query.user_message

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch_dtype)

    input_len = inputs["input_ids"].shape[-1]

    # === Generation ===
    start_time = time.time()
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True
        )
    end_time = time.time()
    elapsed_time = end_time - start_time

    # === Decode & Token Info ===
    full_seq = gen_out.sequences[0]
    generated_ids = full_seq[input_len:]
    output_tokens = len(generated_ids)
    decoded = processor.decode(generated_ids, skip_special_tokens=True)
    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

    # === Token-level logits ===
    token_level_logits = []
    for i, (token_id, score_vec) in enumerate(zip(generated_ids, gen_out.scores)):
        token_str = processor.tokenizer.decode([token_id])
        topk = torch.topk(score_vec.squeeze(), k=5)
        top_tokens = processor.tokenizer.convert_ids_to_tokens(topk.indices.tolist())
        top_logits = [{"token": t, "logit": round(float(s), 4)} for t, s in zip(top_tokens, topk.values)]
        token_level_logits.append({
            "token": token_str,
            "top_logits": top_logits
        })

    # === Compute logprob ===
    with torch.inference_mode():
        full_input_ids = full_seq.unsqueeze(0)
        labels = full_input_ids.clone()
        labels[0, :input_len] = -100
        output = model(input_ids=full_input_ids.to(model.device), labels=labels.to(model.device))
        logprob = -output.loss.item() * output_tokens

    return {
        "response": decoded,
        "elapsed_time": round(elapsed_time, 2),
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "logprob": round(logprob, 4),
        "token_level_logits": token_level_logits
    }
