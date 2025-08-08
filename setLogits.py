import requests
import time
import math

def get_logprob(answer_text):
    """Send a prompt with a specific answer and get log-probability from model."""
    prompt = (
        "Question: What gas do plants release during photosynthesis?\n"
        "Choices: (A) Oxygen (B) Carbon Dioxide (C) Nitrogen (D) Methane\n"
        "Answer: "
    )
    full_input = prompt + answer_text

    try:
        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"user_message": full_input}
        )
        response.raise_for_status()
        result = response.json()

        if "logprob" not in result:
            print(f"[Warning] No logprob found for answer '{answer_text}', using fallback.")
            return -len(answer_text) * 1.0  # fallback score

        return result["logprob"]

    except Exception as e:
        print(f"[Error] Failed to get logprob for '{answer_text}': {e}")
        return -len(answer_text) * 1.0  # fallback

if __name__ == '__main__':
    start_time = time.time()

    options = {
        "A": "Oxygen",
        "B": "Carbon Dioxide",
        "C": "Nitrogen",
        "D": "Methane"
    }

    print("==== Getting logprobs for each answer ====")
    logprobs = {}
    for key, ans in options.items():
        logprob = get_logprob(ans)
        logprobs[key] = logprob
        print(f"{key}. {ans} => logprob: {logprob:.4f}")

    # Convert logprobs to softmax probabilities
    max_logprob = max(logprobs.values())  # for numerical stability
    unnormalized = {k: math.exp(v - max_logprob) for k, v in logprobs.items()}
    total = sum(unnormalized.values()) or 1e-8  # avoid divide-by-zero
    confidences = {k: v / total for k, v in unnormalized.items()}

    print("\n==== Normalized Confidence Scores ====")
    for key, conf in sorted(confidences.items(), key=lambda x: -x[1]):
        print(f"{key}. {options[key]} => confidence: {conf:.2%}")

    end_time = time.time()
    print(f"\nTotal local request time: {end_time - start_time:.2f} seconds")
