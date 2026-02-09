#!/bin/bash
# Create demo adapter directories for testing

ADAPTERS_DIR="adapters"

echo "Creating demo adapter directories..."

# Create adapter directories
for i in {1..4}; do
    adapter_dir="${ADAPTERS_DIR}/adapter${i}"
    mkdir -p "$adapter_dir"

    # Create a placeholder file (in real scenario, these would be actual LoRA weights)
    echo "# Placeholder for adapter${i}" > "$adapter_dir/README.md"
    echo "This is a placeholder. Replace with actual LoRA adapter files (.safetensors or .bin)" >> "$adapter_dir/README.md"

    echo "Created: $adapter_dir"
done

echo ""
echo "Demo adapter structure created!"
echo "Replace placeholder files with actual LoRA adapters."
echo ""
echo "Expected structure for each adapter:"
echo "  adapters/adapter1/"
echo "    ├── adapter_config.json"
echo "    └── adapter_model.safetensors (or .bin)"
