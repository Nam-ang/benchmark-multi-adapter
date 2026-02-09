# API Documentation

## Configuration File (config.yaml)

### Model Settings
```yaml
model:
  name: str                      # HuggingFace model ID or local path
  tensor_parallel_size: int      # Number of GPUs for tensor parallelism
  gpu_memory_utilization: float  # GPU memory utilization (0.0-1.0)
  max_model_len: int            # Maximum sequence length
```

### Server Settings
```yaml
server:
  host: str  # Server host (default: "localhost")
  port: int  # Server port (default: 8000)
```

### LoRA Settings
```yaml
lora:
  adapters_dir: str      # Directory containing LoRA adapters
  max_loras: int        # Maximum number of LoRAs to load simultaneously
  max_lora_rank: int    # Maximum LoRA rank
  max_cpu_loras: int    # Number of LoRAs to cache in CPU memory
```

### Benchmark Settings
```yaml
benchmark:
  dataset_path: str                    # Path to dataset file (JSONL)
  num_requests: int                    # Total number of requests
  concurrency: int                     # Number of concurrent requests
  shuffle_data: bool                   # Whether to shuffle dataset
  random_adapter_assignment: bool      # Randomly assign adapters to requests
```

### Generation Settings
```yaml
generation:
  max_tokens: int      # Maximum tokens to generate
  temperature: float   # Sampling temperature
  top_p: float        # Nucleus sampling parameter
  top_k: int          # Top-k sampling parameter
```

## Dataset Format

### JSONL Format
Each line should be a JSON object with the following fields:

```json
{
  "prompt": "Your prompt text here",
  "adapter_name": "adapter1"  // Optional, required if random_adapter_assignment is false
}
```

### Example Dataset
```jsonl
{"prompt": "Explain machine learning"}
{"prompt": "What is deep learning?", "adapter_name": "math_adapter"}
{"prompt": "Write a Python function", "adapter_name": "code_adapter"}
```

## Utility Functions

### `load_config(config_path: str) -> Dict`
Load configuration from YAML file.

**Parameters:**
- `config_path`: Path to config file

**Returns:** Configuration dictionary

### `load_dataset(dataset_path: str, shuffle: bool) -> List[Dict]`
Load dataset from JSONL file.

**Parameters:**
- `dataset_path`: Path to JSONL file
- `shuffle`: Whether to shuffle data

**Returns:** List of dataset items

### `get_available_adapters(adapters_dir: str) -> List[str]`
Get list of available LoRA adapters.

**Parameters:**
- `adapters_dir`: Directory containing adapters

**Returns:** List of adapter names

### `assign_random_adapters(data: List[Dict], adapters: List[str]) -> List[Dict]`
Assign random adapters to dataset items.

**Parameters:**
- `data`: Dataset items
- `adapters`: List of adapter names

**Returns:** Dataset with assigned adapters

## Benchmark Class

### `MultiLoRABenchmark`

#### `__init__(config: Dict)`
Initialize benchmark client.

#### `async send_request(session, prompt, adapter_name, request_id) -> Dict`
Send a single generation request.

**Parameters:**
- `session`: aiohttp ClientSession
- `prompt`: Input prompt
- `adapter_name`: LoRA adapter name (optional)
- `request_id`: Unique request ID

**Returns:** Request result dictionary

#### `async run_benchmark(data: List[Dict], concurrency: int)`
Run benchmark with specified concurrency.

**Parameters:**
- `data`: Dataset items
- `concurrency`: Number of concurrent requests

#### `analyze_results() -> Dict`
Analyze benchmark results.

**Returns:** Analysis dictionary with metrics

#### `save_detailed_results(output_dir: str)`
Save detailed results to CSV.

#### `plot_results(output_dir: str)`
Create visualization plots.

## Results Format

### JSON Results
```json
{
  "total_requests": 100,
  "successful_requests": 98,
  "failed_requests": 2,
  "total_duration": 45.6,
  "metrics": {
    "mean_latency": 0.234,
    "p50_latency": 0.220,
    "p95_latency": 0.412,
    "p99_latency": 0.523,
    "requests_per_sec": 2.15,
    "tokens_per_sec": 43.2
  },
  "adapter_stats": {
    "adapter1": {
      "count": 25,
      "avg_latency": 0.240,
      "p95_latency": 0.405
    }
  },
  "config": { ... },
  "timestamp": "2024-02-06T14:30:22"
}
```

### CSV Results
Columns:
- `request_id`: Request ID
- `prompt`: Input prompt
- `adapter_name`: LoRA adapter used
- `latency`: Request latency (seconds)
- `num_tokens`: Number of tokens generated
- `generated_text`: Generated text
- `error`: Error message (if any)
- `timestamp`: Request timestamp

## Command Line Interface

### Server
```bash
python src/server.py [--config CONFIG_PATH]
```

### Benchmark
```bash
python src/benchmark.py [OPTIONS]

Options:
  --config PATH         Configuration file path (default: config.yaml)
  --dataset PATH        Dataset file path (overrides config)
  --concurrency INT     Number of concurrent requests (overrides config)
  --num-requests INT    Total number of requests (overrides config)
```

## vLLM API Endpoints

### Completions Endpoint
```
POST http://localhost:8000/v1/completions
```

**Request Body:**
```json
{
  "model": "meta-llama/Llama-2-7b-hf",
  "prompt": "Your prompt here",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "lora_request": {
    "lora_name": "adapter1",
    "lora_path": "/path/to/adapter1",
    "lora_int_id": 1
  }
}
```

**Response:**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "meta-llama/Llama-2-7b-hf",
  "choices": [{
    "text": "Generated text...",
    "index": 0,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```
