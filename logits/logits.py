import requests
import json
import time

# === API endpoint ===
API_URL = "http://localhost:8000/chat"

# === Fixed question stem ===
base_question = (
    "Q: Let A = {x ∈ ℝ | |x| ≤ 2}, B = {x ∈ ℤ | √x ≤ 4}. Then A ∩ B = ?\n"
)

# === Option set ===
options = {
    "A": "(0,2)",
    "B": "[0,2]",
    "C": "{0,2}",
    "D": "{0,1,2}"
}

# === Construct full prompt and call API ===
def call_api(prompt):
    payload = {"user_message": prompt}
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"[ERROR] API call failed: {e}")
        return None


def task2_true_false_with_context():
    print("=== Task 2: Evaluate Each Option with Full Context, Expect True/False Only ===")
    results = {}

    for key, choice in options.items():
        # Construct prompt
        prompt = (
            base_question +
            f"{key}. {choice}\n"
            "Answer with only one word: True or False.\n"
            "Answer:"
        )

        print(f"\n>>> Prompt [{key}]:\n{prompt}")
        result = call_api(prompt)

        entry = {
            "prompt": prompt,
            "response": None,
            "logits": {},
            "token_level_logits": [],
            "raw": result
        }

        if result:
            entry["response"] = result.get("response", "")
            entry["token_level_logits"] = result.get("token_level_logits", [])

            # Extract logits of True / False (from token-level logits)
            for step in entry["token_level_logits"]:
                tok = step.get("token", "")
                if tok in ["True", "False"]:
                    for top in step.get("top_logits", []):
                        if top["token"] == tok:
                            entry["logits"][tok] = top["logit"]

        results[key] = entry
        time.sleep(1)  # Optional: adjust or remove delay if needed

    # Save results to unified JSON file
    with open("results_task2.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Saved to results_task2.json")


if __name__ == "__main__":
    task2_true_false_with_context()
