import time
import psutil
import ollama

prompt = "Explain Retrieval Augmented Generation for digital publications."

print("\nRunning benchmark...\n")

start_time = time.time()
first_token_time = None

response_text = ""

stream = ollama.chat(
    model='qwen2.5:7b',
    messages=[
        {'role': 'user', 'content': prompt}
    ],
    stream=True
)

for chunk in stream:
    if first_token_time is None:
        first_token_time = time.time()

    token = chunk['message']['content']
    response_text += token
    print(token, end='', flush=True)

end_time = time.time()

print("\n\nPERFORMANCE METRICS")
print("-------------------")

ttft = first_token_time - start_time
total_latency = end_time - start_time

print(f"TTFT (Time To First Token): {ttft:.2f} seconds")
print(f"Total Latency: {total_latency:.2f} seconds")

if total_latency > ttft:
    generated_time = total_latency - ttft
    tokens_estimate = len(response_text.split())

    print(f"Estimated Tokens/sec: {tokens_estimate / generated_time:.2f}")

print(f"RAM Usage: {psutil.virtual_memory().used / 1e9:.2f} GB")
print(f"CPU Usage: {psutil.cpu_percent(interval=1)} %")