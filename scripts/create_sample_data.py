#!/usr/bin/env python3
"""Create sample dataset for testing."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import create_sample_dataset


def main():
    """Create sample dataset."""
    output_path = "data/benchmark_dataset.jsonl"
    num_samples = 100

    print(f"Creating sample dataset with {num_samples} samples...")
    create_sample_dataset(output_path, num_samples)
    print("Done!")


if __name__ == "__main__":
    main()
