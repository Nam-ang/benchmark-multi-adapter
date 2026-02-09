"""vLLM server with multi-LoRA support."""

import argparse
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.entrypoints.openai.api_server import run_server
import yaml


def start_vllm_server(config_path: str = "config.yaml"):
    """Start vLLM server with multi-LoRA support."""
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    server_config = config['server']
    lora_config = config['lora']

    # Get adapter paths
    adapters_dir = Path(lora_config['adapters_dir'])
    adapter_paths = []

    if adapters_dir.exists():
        for adapter_dir in adapters_dir.iterdir():
            if adapter_dir.is_dir():
                adapter_paths.append(str(adapter_dir))

    print(f"Found {len(adapter_paths)} LoRA adapters")
    for path in adapter_paths:
        print(f"  - {Path(path).name}")

    # Start vLLM OpenAI-compatible server
    print(f"\nStarting vLLM server on {server_config['host']}:{server_config['port']}")
    print(f"Base model: {model_config['name']}")
    print(f"Max LoRAs: {lora_config['max_loras']}")
    print(f"Max CPU LoRAs: {lora_config['max_cpu_loras']}")

    # Build vLLM arguments
    args = [
        "--model", model_config['name'],
        "--host", server_config['host'],
        "--port", str(server_config['port']),
        "--enable-lora",
        "--max-loras", str(lora_config['max_loras']),
        "--max-lora-rank", str(lora_config['max_lora_rank']),
        "--max-cpu-loras", str(lora_config['max_cpu_loras']),
        "--tensor-parallel-size", str(model_config['tensor_parallel_size']),
        "--gpu-memory-utilization", str(model_config['gpu_memory_utilization']),
        "--max-model-len", str(model_config['max_model_len']),
    ]

    # Note: This is a simplified version. In practice, you'd use:
    # python -m vllm.entrypoints.openai.api_server <args>
    print("\nTo start the server, run:")
    print(f"python -m vllm.entrypoints.openai.api_server {' '.join(args)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start vLLM server with multi-LoRA support"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()
    start_vllm_server(args.config)


if __name__ == "__main__":
    main()
