import json
import requests
import json
import time
import re
# === API endpoint ===
API_URL = "http://localhost:8000/chat"

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
            "Answer with only one word: True or False. Do not use any additional reasoning or justification\n"
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
            def normalize(tok):
               # First remove markdown formatting (bold, italic, etc.)
               tok = re.sub(r'\*\*|\*|__|_', '', tok)  # Remove **, *, __, _
               # Then remove non-alphabetic characters from start/end
               return re.sub(r"^[^A-Za-z]*|[^A-Za-z]*$", "", tok).strip()

            for step in reversed(entry["token_level_logits"]):
                    tok = step.get("token", "")
                    if normalize(tok).upper() in ["TRUE", "FALSE"]:
                        for top in step.get("top_logits", []):
                            if normalize(top["token"]).upper() == normalize(tok).upper():
                                entry["logits"][normalize(tok).upper()] = top["logit"]
                                break
                        break

        results[key] = entry
        
        time.sleep(1)  # Optional: adjust or remove delay if needed

    full_results.append(results)
    print("\n✅ Saved to full_results")


full_results = []

original_dataset = json.load(open("/blue/iruchkin/a.venkat/llms/original_dataset.json"))

for item in original_dataset:
    idx_a = len(item['question_en']) - item['question_en'][::-1].index('.A')
    idx_b = len(item['question_en']) - item['question_en'][::-1].index('.B')
    idx_c = len(item['question_en']) - item['question_en'][::-1].index('.C')
    idx_d = len(item['question_en']) - item['question_en'][::-1].index('.D')

    choice_A = item['question_en'][idx_a : idx_b-2]
    choice_B = item['question_en'][idx_b : idx_c-2]
    choice_C = item['question_en'][idx_c : idx_d-2]
    choice_D = item['question_en'][idx_d : ]

    options = {
    "A": choice_A,
    "B": choice_B,
    "C": choice_C,
    "D": choice_D
}
    base_question = item['question_en'][:idx_a-2]
    
    task2_true_false_with_context()


with open("full_results_task2.json", "w") as f:
    json.dump(full_results, f, indent=2)

print("\n✅ Saved to full_results_task2.json")