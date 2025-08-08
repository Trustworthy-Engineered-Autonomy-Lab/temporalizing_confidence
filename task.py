import requests
import time

if __name__ == '__main__':
    start_time = time.time()  # Start time

    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"user_message": "Hello, please introduce Gemma 3."}
    )

    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Local request duration

    # Parse the JSON returned from the server
    result = response.json()

    print("==== Response Info ====")
    print(f"Model response: {result['response']}")
    print(f"Inference time on server: {result['elapsed_time']} seconds")
    print(f"Number of input tokens: {result['input_tokens']}")
    print(f"Number of output tokens: {result['output_tokens']}")
    print(f"Output speed: {result['tokens_per_second']} tokens/sec")
    print("------------------------")
    print(f"Total local request time: {elapsed_time:.2f} seconds")

