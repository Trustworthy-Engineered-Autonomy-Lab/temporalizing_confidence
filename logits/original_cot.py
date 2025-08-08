import json
import requests
import json
import time
from logits import call_api
# === API endpoint ===
API_URL = "http://localhost:8000/chat"

def task2_true_false_with_context():
    print("=== Task 2: Evaluate Each Option with Full Context, Expect True/False Only ===")
    results = {}

    for key, choice in options.items():
        # Construct prompt
        prompt = (
            base_question +
            f"{key}. {choice}\n"
            "Answer with True or False. Use chain of thought reasoning to justify your answer.\n"
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

            for step in reversed(entry["token_level_logits"]):
                    tok = step.get("token", "")
                    if tok in ["True", "False"]:
                        i+=1
                        for top in step.get("top_logits", []):
                            if top["token"] == tok:
                                entry["logits"][tok] = top["logit"]
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


with open("CoT_full_results_task2.json", "w") as f: #Change name to match model
    json.dump(full_results, f, indent=2)

print("\n✅ Saved to full_results_task2.json")