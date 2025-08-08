from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
import time

# === Initialize model ===
model_id = "google/gemma-3-12b-it"

torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="balanced",  # <<<<<<--- Changed to 'balanced' for automatic GPU distribution
    torch_dtype=torch_dtype
).eval()

print(f"Model main device: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Name 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Name 1: {torch.cuda.get_device_name(1) if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 'None'}")

processor = AutoProcessor.from_pretrained(model_id)

# === Create FastAPI app ===
app = FastAPI()

# === Define input/output schema ===
class QueryRequest(BaseModel):
    user_message: str

class QueryResponse(BaseModel):
    response: str
    elapsed_time: float
    input_tokens: int
    output_tokens: int
    tokens_per_second: float

# === Define API route ===
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(query: QueryRequest):
    user_text = query.user_message

    # Construct messages according to Gemma 3 format
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

    # Preprocess input
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch_dtype)

    input_len = inputs["input_ids"].shape[-1]

    # Start timing
    start_time = time.time()

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            use_cache=True  # Ensure caching is enabled for faster inference
        )
        generation = generation[0][input_len:]

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Decode generated tokens
    decoded = processor.decode(generation, skip_special_tokens=True)

    # Count number of output tokens
    output_tokens = generation.shape[-1]

    # Calculate tokens per second
    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

    return QueryResponse(
        response=decoded,
        elapsed_time=round(elapsed_time, 2),
        input_tokens=input_len,
        output_tokens=output_tokens,
        tokens_per_second=round(tokens_per_second, 2)
    )
