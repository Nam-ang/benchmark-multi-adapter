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
from transformers import AutoTokenizer

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

    def __init__(self, config: Dict[str, Any], mode: str = 'multi'):
        """
        Initialize benchmark client.

        Args:
            config: Configuration dictionary
            mode: Benchmark mode - 'multi' for separate adapters per task,
                  'baseline' for single multi-task adapter
        """
        self.config = config
        self.mode = mode
        self.server_url = f"http://{config['server']['host']}:{config['server']['port']}"
        self.results = []
        self.adapter_stats = defaultdict(lambda: {'count': 0, 'latencies': []})

        # Load tokenizer for chat template
        print(f"Loading tokenizer: {config['model']['name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name'],
            trust_remote_code=True
        )

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

        # Apply chat template to the prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare request payload
        payload = {
            "model": self.config['model']['name'],
            "prompt": formatted_prompt,
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
            'formatted_prompt': formatted_prompt,
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
        mode_str = "baseline (single multi-task adapter)" if self.mode == 'baseline' else "multi-adapter"
        print(f"\nRunning {mode_str} benchmark with {len(data)} requests, concurrency={concurrency}")

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
            'mode': self.mode,
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


def print_comparison(baseline: Dict[str, Any], multi: Dict[str, Any]):
    """Print comparison between baseline and multi-adapter results."""
    baseline_metrics = baseline['metrics']
    multi_metrics = multi['metrics']

    def calc_improvement(baseline_val, multi_val, lower_is_better=True):
        if baseline_val == 0:
            return 0
        diff = ((baseline_val - multi_val) / baseline_val) * 100
        return diff if lower_is_better else -diff

    print("\n--- Latency Comparison ---")
    print(f"{'Metric':<20} {'Baseline':<15} {'Multi-Adapter':<15} {'Improvement':<15}")
    print("-" * 65)

    metrics_to_compare = [
        ('mean_latency', 'Mean Latency', True),
        ('p50_latency', 'P50 Latency', True),
        ('p95_latency', 'P95 Latency', True),
        ('p99_latency', 'P99 Latency', True),
        ('requests_per_sec', 'Requests/sec', False),
        ('tokens_per_sec', 'Tokens/sec', False),
    ]

    for key, label, lower_is_better in metrics_to_compare:
        baseline_val = baseline_metrics.get(key, 0)
        multi_val = multi_metrics.get(key, 0)
        improvement = calc_improvement(baseline_val, multi_val, lower_is_better)

        if 'latency' in key:
            baseline_str = f"{baseline_val:.3f}s"
            multi_str = f"{multi_val:.3f}s"
        else:
            baseline_str = f"{baseline_val:.2f}"
            multi_str = f"{multi_val:.2f}"

        improvement_str = f"{improvement:+.2f}%"
        print(f"{label:<20} {baseline_str:<15} {multi_str:<15} {improvement_str:<15}")

    print("\n--- Request Statistics ---")
    print(f"{'Metric':<30} {'Baseline':<15} {'Multi-Adapter':<15}")
    print("-" * 60)
    print(f"{'Total Requests':<30} {baseline['total_requests']:<15} {multi['total_requests']:<15}")
    print(f"{'Successful Requests':<30} {baseline['successful_requests']:<15} {multi['successful_requests']:<15}")
    print(f"{'Failed Requests':<30} {baseline['failed_requests']:<15} {multi['failed_requests']:<15}")
    print(f"{'Total Duration':<30} {baseline['total_duration']:.2f}s{'':<10} {multi['total_duration']:.2f}s{'':<10}")


def save_comparison(
    baseline: Dict[str, Any],
    multi: Dict[str, Any],
    output_dir: str
):
    """Save comparison results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{timestamp}.json"
    filepath = output_path / filename

    comparison = {
        'baseline': baseline,
        'multi_adapter': multi,
        'timestamp': datetime.now().isoformat(),
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\nComparison results saved to: {filepath}")
    return filepath


def plot_comparison(
    baseline: Dict[str, Any],
    multi: Dict[str, Any],
    output_dir: str
):
    """Create comparison visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_path / f"comparison_{timestamp}_plot.png"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Baseline vs Multi-Adapter Comparison', fontsize=16, fontweight='bold')

    baseline_metrics = baseline['metrics']
    multi_metrics = multi['metrics']

    # 1. Latency comparison
    latency_metrics = ['mean_latency', 'p50_latency', 'p95_latency', 'p99_latency']
    latency_labels = ['Mean', 'P50', 'P95', 'P99']

    baseline_latencies = [baseline_metrics.get(m, 0) for m in latency_metrics]
    multi_latencies = [multi_metrics.get(m, 0) for m in latency_metrics]

    x = range(len(latency_labels))
    width = 0.35

    axes[0, 0].bar([i - width/2 for i in x], baseline_latencies, width, label='Baseline', alpha=0.8)
    axes[0, 0].bar([i + width/2 for i in x], multi_latencies, width, label='Multi-Adapter', alpha=0.8)
    axes[0, 0].set_xlabel('Metric')
    axes[0, 0].set_ylabel('Latency (s)')
    axes[0, 0].set_title('Latency Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(latency_labels)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Throughput comparison
    throughput_metrics = ['requests_per_sec', 'tokens_per_sec']
    throughput_labels = ['Requests/sec', 'Tokens/sec']

    baseline_throughput = [baseline_metrics.get(m, 0) for m in throughput_metrics]
    multi_throughput = [multi_metrics.get(m, 0) for m in throughput_metrics]

    x = range(len(throughput_labels))
    axes[0, 1].bar([i - width/2 for i in x], baseline_throughput, width, label='Baseline', alpha=0.8)
    axes[0, 1].bar([i + width/2 for i in x], multi_throughput, width, label='Multi-Adapter', alpha=0.8)
    axes[0, 1].set_xlabel('Metric')
    axes[0, 1].set_ylabel('Throughput')
    axes[0, 1].set_title('Throughput Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(throughput_labels)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Success rate comparison
    baseline_success_rate = (baseline['successful_requests'] / baseline['total_requests'] * 100
                             if baseline['total_requests'] > 0 else 0)
    multi_success_rate = (multi['successful_requests'] / multi['total_requests'] * 100
                          if multi['total_requests'] > 0 else 0)

    axes[0, 2].bar(['Baseline', 'Multi-Adapter'],
                   [baseline_success_rate, multi_success_rate],
                   alpha=0.8, color=['#1f77b4', '#ff7f0e'])
    axes[0, 2].set_ylabel('Success Rate (%)')
    axes[0, 2].set_title('Success Rate Comparison')
    axes[0, 2].set_ylim([0, 105])
    axes[0, 2].grid(True, alpha=0.3)

    for i, v in enumerate([baseline_success_rate, multi_success_rate]):
        axes[0, 2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

    # 4. Improvement percentage
    improvements = {}
    for metric in latency_metrics:
        baseline_val = baseline_metrics.get(metric, 0)
        multi_val = multi_metrics.get(metric, 0)
        if baseline_val > 0:
            improvements[metric] = ((baseline_val - multi_val) / baseline_val) * 100

    if improvements:
        axes[1, 0].barh(list(range(len(improvements))), list(improvements.values()), alpha=0.8)
        axes[1, 0].set_yticks(range(len(improvements)))
        axes[1, 0].set_yticklabels([m.replace('_', ' ').title() for m in improvements.keys()])
        axes[1, 0].set_xlabel('Improvement (%)')
        axes[1, 0].set_title('Latency Improvement (Baseline → Multi-Adapter)')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 5. Adapter usage (multi-adapter only)
    if 'adapter_stats' in multi and multi['adapter_stats']:
        adapter_names = list(multi['adapter_stats'].keys())
        adapter_counts = [stats['count'] for stats in multi['adapter_stats'].values()]

        axes[1, 1].bar(range(len(adapter_names)), adapter_counts, alpha=0.8)
        axes[1, 1].set_xticks(range(len(adapter_names)))
        axes[1, 1].set_xticklabels(adapter_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Request Count')
        axes[1, 1].set_title('Multi-Adapter: Requests per Adapter')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No adapter stats available',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Multi-Adapter: Requests per Adapter')

    # 6. Summary statistics table
    axes[1, 2].axis('off')
    summary_data = [
        ['Metric', 'Baseline', 'Multi-Adapter'],
        ['Total Duration', f"{baseline['total_duration']:.2f}s", f"{multi['total_duration']:.2f}s"],
        ['Successful Req', str(baseline['successful_requests']), str(multi['successful_requests'])],
        ['Failed Req', str(baseline['failed_requests']), str(multi['failed_requests'])],
        ['Mean Latency', f"{baseline_metrics['mean_latency']:.3f}s", f"{multi_metrics['mean_latency']:.3f}s"],
        ['P95 Latency', f"{baseline_metrics['p95_latency']:.3f}s", f"{multi_metrics['p95_latency']:.3f}s"],
    ]

    table = axes[1, 2].table(cellText=summary_data, cellLoc='center', loc='center',
                            colWidths=[0.35, 0.325, 0.325])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    axes[1, 2].set_title('Summary Statistics')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


async def run_single_benchmark(
    config: Dict[str, Any],
    mode: str,
    dataset_path: str,
    concurrency: int,
    num_requests: int,
) -> Dict[str, Any]:
    """Run a single benchmark experiment."""
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    data = load_dataset(
        dataset_path,
        shuffle=config['benchmark']['shuffle_data']
    )

    # Limit to num_requests
    data = data[:num_requests]

    # Get available adapters
    adapters = get_available_adapters(config['lora']['adapters_dir'])
    print(f"Found {len(adapters)} adapters: {adapters}")

    # Assign adapters based on mode
    if mode == 'baseline':
        # Use single multi-task adapter for all requests
        baseline_adapter = config['benchmark'].get('baseline_adapter', 'multi_task_adapter')
        print(f"Using baseline adapter '{baseline_adapter}' for all requests")
        for item in data:
            item['adapter_name'] = baseline_adapter
    elif mode == 'multi':
        # Check if adapter_name already exists in data (to maintain pairing)
        has_adapters_in_data = all('adapter_name' in item for item in data)

        if has_adapters_in_data:
            print("Using adapter_name from dataset (maintaining prompt-adapter pairing)")
            adapter_counts = {}
            for item in data:
                adapter = item['adapter_name']
                adapter_counts[adapter] = adapter_counts.get(adapter, 0) + 1
            print(f"Adapter distribution: {adapter_counts}")
        elif config['benchmark']['random_adapter_assignment'] and adapters:
            print("⚠️  Warning: Assigning random adapters (may break prompt-adapter pairing)")
            data = assign_random_adapters(data, adapters)
        elif not adapters:
            print("Warning: No adapters found, running without LoRA")

    # Run benchmark
    benchmark = MultiLoRABenchmark(config, mode=mode)

    start_time = time.time()
    await benchmark.run_benchmark(data, concurrency)
    end_time = time.time()

    print(f"\n{mode.upper()} benchmark completed in {end_time - start_time:.2f}s")

    # Analyze results
    analysis = benchmark.analyze_results()

    # Print summary
    print(format_results_summary(analysis))

    # Save results
    output_dir = config['output']['results_dir']
    results_path = save_results(analysis, output_dir, mode=mode)

    if config['output']['save_detailed_logs']:
        benchmark.save_detailed_results(output_dir)

    return analysis


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
    parser.add_argument(
        "--mode",
        type=str,
        choices=['baseline', 'multi', 'compare'],
        default='multi',
        help="Benchmark mode: baseline (single adapter), multi (multiple adapters), or compare (run both)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    dataset_path = args.dataset or config['benchmark']['dataset_path']
    concurrency = args.concurrency or config['benchmark']['concurrency']
    num_requests = args.num_requests or config['benchmark']['num_requests']

    if args.mode == 'compare':
        # Run both baseline and multi-adapter benchmarks
        print("\n" + "="*60)
        print("RUNNING COMPARISON BENCHMARK")
        print("="*60)

        baseline_results = await run_single_benchmark(
            config, 'baseline', dataset_path, concurrency, num_requests
        )

        multi_results = await run_single_benchmark(
            config, 'multi', dataset_path, concurrency, num_requests
        )

        # Generate comparison report
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print_comparison(baseline_results, multi_results)

        # Save comparison results
        output_dir = config['output']['results_dir']
        save_comparison(baseline_results, multi_results, output_dir)

        # Plot comparison
        if config['output']['plot_results']:
            plot_comparison(baseline_results, multi_results, output_dir)

    else:
        # Run single benchmark
        await run_single_benchmark(
            config, args.mode, dataset_path, concurrency, num_requests
        )


if __name__ == "__main__":
    asyncio.run(main())
