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
    device_map="auto",  # <<<<<<--- Changed to 'auto' to use only one GPU
    torch_dtype=torch_dtype
).eval()

print(f"Model main device: {next(model.parameters()).device}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Name 0: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"GPU Count: {torch.cuda.device_count()}")

processor = AutoProcessor.from_pretrained(model_id)

# === Create FastAPI application ===
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

# === API route ===
@app.post("/chat", response_model=QueryResponse)
@app.post("/chat")
async def chat_endpoint(query: QueryRequest):
    user_text = query.user_message

    # Construct messages for Gemma-3
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
            use_cache=True
        )
        generation = generation[0][input_len:]

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Decode output
    decoded = processor.decode(generation, skip_special_tokens=True)
    output_tokens = generation.shape[-1]
    tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0

    # Calculate log-prob using the model again with labels
    with torch.inference_mode():
        # Reconstruct full input + generated tokens for logprob computation
        full_input_ids = torch.cat([inputs["input_ids"][0], generation], dim=0).unsqueeze(0)
        full_inputs = {"input_ids": full_input_ids.to(model.device)}
        labels = full_input_ids.clone()

        # Replace the input portion of the labels with -100 so they are ignored
        labels[0, :input_len] = -100

        # Compute log-probability via negative loss
        output = model(**full_inputs, labels=labels)
        logprob = -output.loss.item() * output_tokens

    # Return full response including logprob
    return {
        "response": decoded,
        "elapsed_time": round(elapsed_time, 2),
        "input_tokens": input_len,
        "output_tokens": output_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "logprob": round(logprob, 4)  # new field!
    }

