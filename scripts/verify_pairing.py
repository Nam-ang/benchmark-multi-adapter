"""Verify that prompt-adapter pairing is maintained during shuffle."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_dataset


def verify_pairing():
    """Verify that prompt-adapter pairs remain consistent across shuffles."""

    print("="*70)
    print("PROMPT-ADAPTER PAIRING VERIFICATION")
    print("="*70)

    dataset_path = "examples/example_dataset.jsonl"

    # Load without shuffle to see original order
    print("\n1. Loading dataset WITHOUT shuffle (original order):")
    data_original = load_dataset(dataset_path, shuffle=False)

    print(f"   Total items: {len(data_original)}")
    print(f"\n   {'Index':<8} {'Adapter':<20} {'Prompt (first 60 chars)'}")
    print("   " + "-"*95)
    for i, item in enumerate(data_original[:5]):
        adapter = item.get('adapter_name', 'MISSING')
        prompt = item['prompt'][:60]
        print(f"   {i:<8} {adapter:<20} {prompt}...")

    # Create a mapping of prompt -> adapter from original data
    prompt_to_adapter = {item['prompt']: item.get('adapter_name') for item in data_original}

    # Load with shuffle multiple times
    print("\n2. Loading dataset WITH shuffle (Run 1):")
    data_shuffle_1 = load_dataset(dataset_path, shuffle=True)

    print(f"   {'Index':<8} {'Adapter':<20} {'Prompt (first 60 chars)'}")
    print("   " + "-"*95)
    for i, item in enumerate(data_shuffle_1[:5]):
        adapter = item.get('adapter_name', 'MISSING')
        prompt = item['prompt'][:60]
        print(f"   {i:<8} {adapter:<20} {prompt}...")

    # Verify pairing is maintained
    print("\n3. Verifying pairing consistency:")
    mismatches = 0
    for item in data_shuffle_1:
        prompt = item['prompt']
        adapter = item.get('adapter_name')
        expected_adapter = prompt_to_adapter.get(prompt)

        if adapter != expected_adapter:
            mismatches += 1
            print(f"   ❌ MISMATCH: prompt='{prompt[:40]}...'")
            print(f"      Expected: {expected_adapter}, Got: {adapter}")

    if mismatches == 0:
        print("   ✅ All pairs maintained correctly!")
    else:
        print(f"   ❌ Found {mismatches} mismatches!")

    # Load with shuffle again (Run 2)
    print("\n4. Loading dataset WITH shuffle (Run 2):")
    data_shuffle_2 = load_dataset(dataset_path, shuffle=True)

    print(f"   {'Index':<8} {'Adapter':<20} {'Prompt (first 60 chars)'}")
    print("   " + "-"*95)
    for i, item in enumerate(data_shuffle_2[:5]):
        adapter = item.get('adapter_name', 'MISSING')
        prompt = item['prompt'][:60]
        print(f"   {i:<8} {adapter:<20} {prompt}...")

    # Verify pairing is maintained in second run
    mismatches_2 = 0
    for item in data_shuffle_2:
        prompt = item['prompt']
        adapter = item.get('adapter_name')
        expected_adapter = prompt_to_adapter.get(prompt)

        if adapter != expected_adapter:
            mismatches_2 += 1

    print(f"\n5. Second shuffle verification:")
    if mismatches_2 == 0:
        print("   ✅ All pairs maintained correctly in second run!")
    else:
        print(f"   ❌ Found {mismatches_2} mismatches in second run!")

    # Summary
    print("\n6. Summary:")
    print(f"   Dataset size: {len(data_original)}")
    print(f"   Unique adapters: {len(set(prompt_to_adapter.values()))}")
    print(f"   Adapter distribution:")
    adapter_counts = {}
    for adapter in prompt_to_adapter.values():
        adapter_counts[adapter] = adapter_counts.get(adapter, 0) + 1
    for adapter, count in sorted(adapter_counts.items()):
        print(f"      {adapter}: {count} prompts")

    print(f"\n   Order comparison (first 5):")
    print(f"   {'Original':<30} vs {'Shuffled':<30}")
    print("   " + "-"*60)
    for i in range(min(5, len(data_original))):
        orig_adapter = data_original[i].get('adapter_name', 'N/A')
        shuf_adapter = data_shuffle_1[i].get('adapter_name', 'N/A')
        match = "✓" if orig_adapter == shuf_adapter else "✗"
        print(f"   {orig_adapter:<30} vs {shuf_adapter:<30} {match}")

    print("\n   📝 Note: Order should be DIFFERENT but pairs should be MAINTAINED")
    print("   i.e., the same prompt always has the same adapter, just in different positions")

    print("\n" + "="*70)


if __name__ == "__main__":
    verify_pairing()
