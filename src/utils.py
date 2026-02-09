"""Utility functions for multi-LoRA benchmark."""

import json
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str, shuffle: bool = True) -> List[Dict[str, Any]]:
    """Load dataset from JSONL file."""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if shuffle:
        random.shuffle(data)

    return data


def get_available_adapters(adapters_dir: str) -> List[str]:
    """Get list of available LoRA adapters."""
    adapters_path = Path(adapters_dir)
    if not adapters_path.exists():
        return []

    adapters = []
    for adapter_dir in adapters_path.iterdir():
        if adapter_dir.is_dir():
            # Check if adapter files exist
            if list(adapter_dir.glob("*.safetensors")) or list(adapter_dir.glob("adapter_*.bin")):
                adapters.append(adapter_dir.name)

    return sorted(adapters)


def assign_random_adapters(
    data: List[Dict[str, Any]],
    adapters: List[str]
) -> List[Dict[str, Any]]:
    """Assign random adapters to each data instance."""
    if not adapters:
        raise ValueError("No adapters available")

    for item in data:
        item['adapter_name'] = random.choice(adapters)

    return data


def save_results(results: Dict[str, Any], output_dir: str = "results"):
    """Save benchmark results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filepath}")
    return filepath


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile value."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def format_results_summary(results: Dict[str, Any]) -> str:
    """Format results summary for display."""
    summary = []
    summary.append("\n" + "="*60)
    summary.append("BENCHMARK RESULTS SUMMARY")
    summary.append("="*60)

    metrics = results.get('metrics', {})

    summary.append(f"\nTotal Requests: {results.get('total_requests', 0)}")
    summary.append(f"Successful Requests: {results.get('successful_requests', 0)}")
    summary.append(f"Failed Requests: {results.get('failed_requests', 0)}")
    summary.append(f"Total Duration: {results.get('total_duration', 0):.2f}s")

    summary.append("\n--- Latency Metrics ---")
    summary.append(f"Mean Latency: {metrics.get('mean_latency', 0):.3f}s")
    summary.append(f"P50 Latency: {metrics.get('p50_latency', 0):.3f}s")
    summary.append(f"P95 Latency: {metrics.get('p95_latency', 0):.3f}s")
    summary.append(f"P99 Latency: {metrics.get('p99_latency', 0):.3f}s")

    summary.append("\n--- Throughput Metrics ---")
    summary.append(f"Requests/sec: {metrics.get('requests_per_sec', 0):.2f}")
    summary.append(f"Tokens/sec: {metrics.get('tokens_per_sec', 0):.2f}")

    if 'adapter_stats' in results:
        summary.append("\n--- Adapter Statistics ---")
        for adapter, stats in results['adapter_stats'].items():
            summary.append(f"{adapter}: {stats['count']} requests, "
                          f"avg latency: {stats['avg_latency']:.3f}s")

    summary.append("="*60 + "\n")

    return "\n".join(summary)


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """Create a sample dataset for testing."""
    prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of using LoRA adapters?",
        "Write a Python function to calculate fibonacci numbers.",
        "Describe the architecture of a transformer model.",
        "How does attention mechanism work in neural networks?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain gradient descent optimization.",
        "What are the advantages of using pre-trained models?",
        "Describe the process of fine-tuning a language model.",
        "How can we reduce overfitting in machine learning models?",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            prompt = random.choice(prompts)
            # Add some variation
            prompt = f"[Query {i+1}] {prompt}"
            data = {"prompt": prompt}
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Sample dataset created: {output_path}")
