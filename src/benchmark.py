"""Multi-LoRA benchmark client."""

import argparse
import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.asyncio import tqdm

from utils import (
    load_config,
    load_dataset,
    get_available_adapters,
    assign_random_adapters,
    save_results,
    calculate_percentile,
    format_results_summary,
)


class MultiLoRABenchmark:
    """Benchmark client for vLLM multi-LoRA."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = f"http://{config['server']['host']}:{config['server']['port']}"
        self.results = []
        self.adapter_stats = defaultdict(lambda: {'count': 0, 'latencies': []})

    async def send_request(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        adapter_name: Optional[str] = None,
        request_id: int = 0,
    ) -> Dict[str, Any]:
        """Send a single generation request."""
        gen_config = self.config['generation']
        lora_config = self.config['lora']

        # Prepare request payload
        payload = {
            "model": self.config['model']['name'],
            "prompt": prompt,
            "max_tokens": gen_config['max_tokens'],
            "temperature": gen_config['temperature'],
            "top_p": gen_config['top_p'],
            "top_k": gen_config['top_k'],
        }

        headers = {"Content-Type": "application/json"}

        # Add LoRA adapter if specified
        if adapter_name:
            # vLLM expects LoRA path in the request
            adapter_path = Path(lora_config['adapters_dir']) / adapter_name
            headers["X-LoRA-Path"] = str(adapter_path)
            # Alternative: some vLLM versions use this format
            payload["lora_request"] = {
                "lora_name": adapter_name,
                "lora_path": str(adapter_path),
                "lora_int_id": request_id,
            }

        start_time = time.time()
        error = None
        response_data = None

        try:
            async with session.post(
                f"{self.server_url}/v1/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    response_data = await response.json()
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
        except Exception as e:
            error = str(e)

        end_time = time.time()
        latency = end_time - start_time

        # Extract generated text and token count
        generated_text = ""
        num_tokens = 0

        if response_data and 'choices' in response_data:
            generated_text = response_data['choices'][0].get('text', '')
            if 'usage' in response_data:
                num_tokens = response_data['usage'].get('completion_tokens', 0)

        result = {
            'request_id': request_id,
            'prompt': prompt,
            'adapter_name': adapter_name,
            'latency': latency,
            'num_tokens': num_tokens,
            'generated_text': generated_text,
            'error': error,
            'timestamp': datetime.now().isoformat(),
        }

        # Update adapter stats
        if adapter_name and not error:
            self.adapter_stats[adapter_name]['count'] += 1
            self.adapter_stats[adapter_name]['latencies'].append(latency)

        return result

    async def run_benchmark(
        self,
        data: List[Dict[str, Any]],
        concurrency: int = 1,
    ):
        """Run benchmark with specified concurrency."""
        print(f"\nRunning benchmark with {len(data)} requests, concurrency={concurrency}")

        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_request(item, req_id):
                async with semaphore:
                    return await self.send_request(
                        session,
                        item['prompt'],
                        item.get('adapter_name'),
                        req_id,
                    )

            # Execute requests with progress bar
            tasks = [
                bounded_request(item, i)
                for i, item in enumerate(data)
            ]

            self.results = []
            for coro in tqdm.as_completed(tasks, total=len(tasks)):
                result = await coro
                self.results.append(result)

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results."""
        if not self.results:
            return {}

        successful_results = [r for r in self.results if not r['error']]
        failed_results = [r for r in self.results if r['error']]

        latencies = [r['latency'] for r in successful_results]
        num_tokens_list = [r['num_tokens'] for r in successful_results]

        total_duration = max([r['latency'] for r in self.results]) if self.results else 0
        total_tokens = sum(num_tokens_list)

        # Calculate metrics
        metrics = {
            'mean_latency': sum(latencies) / len(latencies) if latencies else 0,
            'p50_latency': calculate_percentile(latencies, 50),
            'p95_latency': calculate_percentile(latencies, 95),
            'p99_latency': calculate_percentile(latencies, 99),
            'requests_per_sec': len(successful_results) / total_duration if total_duration > 0 else 0,
            'tokens_per_sec': total_tokens / total_duration if total_duration > 0 else 0,
        }

        # Calculate adapter statistics
        adapter_stats_summary = {}
        for adapter, stats in self.adapter_stats.items():
            if stats['latencies']:
                adapter_stats_summary[adapter] = {
                    'count': stats['count'],
                    'avg_latency': sum(stats['latencies']) / len(stats['latencies']),
                    'p95_latency': calculate_percentile(stats['latencies'], 95),
                }

        analysis = {
            'total_requests': len(self.results),
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'total_duration': total_duration,
            'metrics': metrics,
            'adapter_stats': adapter_stats_summary,
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }

        return analysis

    def save_detailed_results(self, output_dir: str):
        """Save detailed results to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = output_path / f"benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")

        return csv_path

    def plot_results(self, output_dir: str):
        """Create visualization plots."""
        if not self.results:
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = output_path / f"benchmark_{timestamp}_plot.png"

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Multi-LoRA Benchmark Results', fontsize=16)

        successful_results = [r for r in self.results if not r['error']]

        if not successful_results:
            print("No successful results to plot")
            return

        # 1. Latency distribution
        latencies = [r['latency'] for r in successful_results]
        axes[0, 0].hist(latencies, bins=30, edgecolor='black')
        axes[0, 0].set_title('Latency Distribution')
        axes[0, 0].set_xlabel('Latency (s)')
        axes[0, 0].set_ylabel('Frequency')

        # 2. Latency over time
        axes[0, 1].plot(range(len(latencies)), latencies, alpha=0.6)
        axes[0, 1].set_title('Latency Over Time')
        axes[0, 1].set_xlabel('Request Index')
        axes[0, 1].set_ylabel('Latency (s)')

        # 3. Adapter performance comparison
        if self.adapter_stats:
            adapter_names = list(self.adapter_stats.keys())
            avg_latencies = [
                sum(stats['latencies']) / len(stats['latencies'])
                for stats in self.adapter_stats.values()
            ]
            axes[1, 0].bar(adapter_names, avg_latencies)
            axes[1, 0].set_title('Average Latency by Adapter')
            axes[1, 0].set_xlabel('Adapter')
            axes[1, 0].set_ylabel('Avg Latency (s)')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Token throughput
        num_tokens = [r['num_tokens'] for r in successful_results]
        if num_tokens:
            axes[1, 1].scatter(latencies, num_tokens, alpha=0.6)
            axes[1, 1].set_title('Tokens vs Latency')
            axes[1, 1].set_xlabel('Latency (s)')
            axes[1, 1].set_ylabel('Num Tokens')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multi-LoRA benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to dataset (overrides config)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent requests (overrides config)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        help="Number of requests to send (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    dataset_path = args.dataset or config['benchmark']['dataset_path']
    concurrency = args.concurrency or config['benchmark']['concurrency']
    num_requests = args.num_requests or config['benchmark']['num_requests']

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    data = load_dataset(
        dataset_path,
        shuffle=config['benchmark']['shuffle_data']
    )

    # Limit to num_requests
    data = data[:num_requests]

    # Get available adapters
    adapters = get_available_adapters(config['lora']['adapters_dir'])
    print(f"Found {len(adapters)} adapters: {adapters}")

    # Assign random adapters if enabled
    if config['benchmark']['random_adapter_assignment'] and adapters:
        print("Assigning random adapters to requests...")
        data = assign_random_adapters(data, adapters)
    elif not adapters:
        print("Warning: No adapters found, running without LoRA")

    # Run benchmark
    benchmark = MultiLoRABenchmark(config)

    start_time = time.time()
    await benchmark.run_benchmark(data, concurrency)
    end_time = time.time()

    print(f"\nBenchmark completed in {end_time - start_time:.2f}s")

    # Analyze results
    analysis = benchmark.analyze_results()

    # Print summary
    print(format_results_summary(analysis))

    # Save results
    output_dir = config['output']['results_dir']
    results_path = save_results(analysis, output_dir)

    if config['output']['save_detailed_logs']:
        benchmark.save_detailed_results(output_dir)

    if config['output']['plot_results']:
        benchmark.plot_results(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
