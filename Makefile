.PHONY: help install setup-demo clean benchmark server

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup-demo   - Create demo data and adapter structure"
	@echo "  make server       - Show vLLM server start command"
	@echo "  make benchmark    - Run benchmark (server must be running)"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt

setup-demo:
	@echo "Creating sample dataset..."
	@python scripts/create_sample_data.py
	@echo ""
	@echo "Creating demo adapter structure..."
	@bash scripts/setup_demo_adapters.sh

server:
	@python src/server.py

benchmark:
	@python src/benchmark.py

clean:
	@echo "Cleaning generated files..."
	@rm -rf results/*.json results/*.csv results/*.png
	@rm -rf __pycache__ src/__pycache__
	@rm -rf *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"
