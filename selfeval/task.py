import requests
import json

API_URL = "http://localhost:8000/chat"

question_task1 = (
    "Q: Let A = {x ∈ ℝ | |x| ≤ 2}, B = {x ∈ ℤ | √x ≤ 4}. Then A ∩ B = ?\n"
    "A. (0,2)\n"
    "B. [0,2]\n"
    "C. {0,2}\n"
    "D. {0,1,2}\n"
)

def call_api(prompt):
    payload = {"user_message": prompt}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[ERROR] API call failed: {e}")
        return None

def task1_record_full_logits():
    print("=== Task 1: Send full question, record all output token logits ===")
    result = call_api(question_task1)

    if not result:
        print("❌ No result received.")
        return

    output = {
        "prompt": question_task1,
        "response": result.get("response", ""),
        "elapsed_time": result.get("elapsed_time", None),
        "input_tokens": result.get("input_tokens", None),
        "output_tokens": result.get("output_tokens", None),
        "tokens_per_second": result.get("tokens_per_second", None),
        "logprob": result.get("logprob", None),
        "token_level_logits": result.get("token_level_logits", [])  # list of {token, top_logits}
    }

    print("\n🧠 Model Response:")
    print(output["response"])

    print("\n🔍 Token-by-token logits:")
    for step in output["token_level_logits"]:
        print(f"\nGenerated token: '{step['token']}'")
        for top in step["top_logits"]:
            print(f"  {top['token']}: {top['logit']}")

    with open("results_task1.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n✅ Saved to results_task1.json")

if __name__ == "__main__":
    task1_record_full_logits()
