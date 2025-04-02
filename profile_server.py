import subprocess
import json
import random
import time
import threading
import statistics
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

# Configuration
MODEL_PATH = "/home/coder/eon-8b"
FIXED_PREFIX_TOKENS = 1200  # Fixed tokens at the beginning of each request
TOTAL_TOKENS = 1300  # Total tokens per request
OUTPUT_TOKENS = 10
REQUESTS_PER_SECOND = 1  # Adjust as needed
NUM_REQUESTS_TOTAL = 2  # Total number of requests to send
DELIM = "<|end_of_text|>"
DEBUG = False
ENABLE_PROFILING = True  # Toggle for profiling

latencies = []  # Store latencies for analysis

def load_tokenizer(path: str):
    """Loads a tokenizer from the specified path."""
    return PreTrainedTokenizerFast.from_pretrained(path) or PreTrainedTokenizer.from_pretrained(path)

def generate_random_token_ids(tokenizer, n: int):
    """Generates N random token IDs within the tokenizer's vocabulary size."""
    vocab_size = tokenizer.vocab_size
    return [random.randint(0, vocab_size - 1) for _ in range(n)]

def detokenize(tokenizer, token_ids):
    """Detokenizes a list of token IDs into a string."""
    return tokenizer.decode(token_ids)

def random_string(num_tokens: int, tokenizer) -> str:
    """Generates a random string of a given number of tokens."""
    token_ids = generate_random_token_ids(tokenizer, num_tokens)
    return detokenize(tokenizer, token_ids)

def build_request_json(prompt: str) -> str:
    """Builds the JSON request payload."""
    request = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True
    }
    return json.dumps(request)

def send_request(index, request_json, print_responses=False):
    """Sends a single request and times it."""
    print(f"Sending request {index + 1}/{NUM_REQUESTS_TOTAL}")
    start_time = time.time()
    
    process = subprocess.Popen(
        ["curl", "http://localhost:30000/v1/completions", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    stdout, stderr = process.communicate(input=request_json)
    elapsed_time_ms = (time.time() - start_time) * 1000
    latencies.append(elapsed_time_ms)
    print(f"Request {index + 1} completed in {elapsed_time_ms:.2f} ms")
    
    if print_responses and stdout:
        try:
            json_response = json.loads(stdout)
            if DEBUG:
                print("Response:", json.dumps(json_response, indent=4))
        except json.JSONDecodeError:
            print("Invalid JSON response:", stdout)
    if stderr:
        print("Error:", stderr)

def send_requests(requests, requests_per_second, print_responses=False):
    """Sends multiple requests at a controlled rate."""
    if ENABLE_PROFILING:
        print("Starting profiling...")
        subprocess.run("curl -X POST http://localhost:30000/start_profile", shell=True, check=True)
    
    interval = 1.0 / requests_per_second
    threads = []
    
    for i, request_json in enumerate(requests):
        delay = i * interval
        thread = threading.Timer(delay, send_request, args=(i, request_json, print_responses))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    if ENABLE_PROFILING:
        print("Stopping profiling...")
        subprocess.run("curl -X POST http://localhost:30000/stop_profile", shell=True, check=True)
    
    # Print latency statistics
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        median_latency = statistics.median(latencies)
        p99_latency = sorted(latencies)[int(0.99 * len(latencies)) - 1] if len(latencies) > 1 else latencies[0]
        print(f"\nAverage Latency: {avg_latency:.2f} ms")
        print(f"Median Latency: {median_latency:.2f} ms")
        print(f"P99 Latency: {p99_latency:.2f} ms")

# Load tokenizer
tokenizer = load_tokenizer(MODEL_PATH)

# Generate fixed prefix (same for all requests)
fixed_prefix = random_string(FIXED_PREFIX_TOKENS, tokenizer)

# Generate all requests in advance
requests = [
    build_request_json(
        fixed_prefix + DELIM + random_string(TOTAL_TOKENS - FIXED_PREFIX_TOKENS, tokenizer) + DELIM
    )
    for _ in range(NUM_REQUESTS_TOTAL)
]

# Send requests
send_requests(requests, REQUESTS_PER_SECOND, print_responses=True)
