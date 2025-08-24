import json
import httpx
import asyncio
import time
import argparse
import os
import re

# === Parse command line arguments ===
parser = argparse.ArgumentParser(description='Run concurrent CoT evaluation with configurable output path')
parser.add_argument('--output', '-o', type=str, default='CoT_concurrent_results_task2.json', 
                   help='Output file path for results (default: CoT_concurrent_results_task2.json)')
args = parser.parse_args()

# === API endpoint ===
API_URL = "http://localhost:8000/chat"

async def call_api_async(client: httpx.AsyncClient, prompt: str):
    """Async API call function"""
    payload = {"user_message": prompt}
    try:
        response = await client.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"[ERROR] API call failed: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        if hasattr(e, 'response'):
            print(f"[ERROR] Response status: {e.response.status_code}")
            print(f"[ERROR] Response text: {e.response.text}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

async def process_question_async(question_data):
    """Process a single question asynchronously"""
    # Extract question parts
    question_en = question_data['question_en']
    idx_a = len(question_en) - question_en[::-1].index('.A')
    idx_b = len(question_en) - question_en[::-1].index('.B')
    idx_c = len(question_en) - question_en[::-1].index('.C')
    idx_d = len(question_en) - question_en[::-1].index('.D')

    choice_A = question_en[idx_a : idx_b-2]
    choice_B = question_en[idx_b : idx_c-2]
    choice_C = question_en[idx_c : idx_d-2]
    choice_D = question_en[idx_d : ]

    options = {
        "A": choice_A,
        "B": choice_B,
        "C": choice_C,
        "D": choice_D
    }
    base_question = question_en[:idx_a-2]
    
    # Create prompts for all options
    prompts = {}
    for key, choice in options.items():
        prompt = (
            base_question +
            f"{key}. {choice}\n"
            "Answer with True or False. Use chain of thought reasoning to get the answer. Your final answer should be in the format of True or False.\n"
        )
        prompts[key] = prompt

    # Make concurrent API calls for all options
    async with httpx.AsyncClient(timeout=180.0) as client:
        tasks = []
        for key, prompt in prompts.items():
            task = call_api_async(client, prompt)
            tasks.append((key, task))
        
        # Wait for all API calls to complete
        results = {}
        for key, task in tasks:
            print(f"\n>>> Processing [{key}]...")
            result = await task
            
            entry = {
                "prompt": prompts[key],
                "response": None,
                "logits": {},
                "token_level_logits": [],
                "raw": result
            }

            if result:
                entry["response"] = result.get("response", "")
                entry["token_level_logits"] = result.get("token_level_logits", [])

                # Define normalize function for token cleaning
                def normalize(tok):
                    return re.sub(r"^[^A-Za-z]*|[^A-Za-z]*$", "", tok).strip()

                # Extract logits of True / False (from token-level logits)
                for step in reversed(entry["token_level_logits"]):
                    tok = step.get("token", "")
                    if normalize(tok).upper() in ["TRUE", "FALSE"]:
                        for top in step.get("top_logits", []):
                            if normalize(top["token"]).upper() == normalize(tok).upper():
                                entry["logits"][normalize(tok).upper()] = top["logit"]
                                break
                        break

            results[key] = entry

    return results

async def main():
    """Main async function"""
    print("=== Async Task 2: Evaluate Each Option with Full Context ===")
    
    # Load dataset
    original_dataset = json.load(open("/blue/iruchkin/a.venkat/llms/original_dataset.json"))
    
    full_results = []
    
    # Process questions with controlled concurrency
    semaphore = asyncio.Semaphore(5)  # num of questions to process at a time
    
    async def process_with_semaphore(question_data):
        async with semaphore:
            return await process_question_async(question_data)
    
    # Create tasks for all questions
    tasks = [process_with_semaphore(item) for item in original_dataset]
    
    # Process all questions concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # collect results
    for i, result in enumerate(results):
            full_results.append(result)
    
    end_time = time.time()
    print(f"\n✅ Processed {len(full_results)} questions in {end_time - start_time:.2f} seconds") 
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())