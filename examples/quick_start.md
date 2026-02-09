# Quick Start Guide

이 가이드는 vLLM multi-LoRA 벤치마크를 빠르게 시작하는 방법을 설명합니다.

## 1. 환경 설정

### 필수 요구사항
- Python 3.8+
- CUDA 11.8+ (GPU 필요)
- 16GB+ GPU 메모리 권장

### 설치

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 2. 데이터셋 준비

### 옵션 A: 샘플 데이터셋 생성

```bash
python scripts/create_sample_data.py
```

### 옵션 B: 직접 데이터셋 준비

`data/` 디렉토리에 JSONL 파일을 생성하세요:

```bash
mkdir -p data
cat > data/my_dataset.jsonl << 'EOF'
{"prompt": "Explain machine learning"}
{"prompt": "What is deep learning?"}
{"prompt": "Describe neural networks"}
EOF
```

데이터셋 형식:
```json
{"prompt": "질문 또는 프롬프트 텍스트"}
{"prompt": "다른 질문"}
```

## 3. LoRA 어댑터 준비

### 데모 구조 생성

```bash
bash scripts/setup_demo_adapters.sh
```

### 실제 LoRA 어댑터 배치

각 어댑터를 `adapters/` 디렉토리에 배치:

```
adapters/
├── my_adapter_1/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── my_adapter_2/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── my_adapter_3/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**LoRA 어댑터 준비 방법:**

1. **Hugging Face에서 다운로드:**
   ```bash
   # 예시: PEFT로 학습된 모델
   git clone https://huggingface.co/your-username/your-lora-adapter adapters/adapter1
   ```

2. **직접 학습:**
   ```python
   from peft import LoraConfig, get_peft_model, TaskType
   from transformers import AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("base-model")

   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=16,
       lora_alpha=32,
       lora_dropout=0.05,
   )

   model = get_peft_model(model, lora_config)
   # ... 학습 ...
   model.save_pretrained("adapters/my_adapter")
   ```

## 4. 설정 조정

`config.yaml` 파일을 수정하여 설정을 조정하세요:

```yaml
model:
  name: "meta-llama/Llama-2-7b-hf"  # 사용할 베이스 모델

lora:
  max_loras: 4  # 동시 로드할 LoRA 수

benchmark:
  num_requests: 100  # 총 요청 수
  concurrency: 5     # 동시 요청 수
```

## 5. vLLM 서버 시작

### 터미널 1: 서버 실행

```bash
# 서버 시작 명령어 확인
python src/server.py

# 출력된 명령어를 복사하여 실행 (예시):
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --host localhost \
  --port 8000 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --max-cpu-loras 8
```

서버가 시작될 때까지 기다리세요 (모델 로딩에 시간이 걸릴 수 있습니다).

## 6. 벤치마크 실행

### 터미널 2: 벤치마크 실행

```bash
# 기본 설정으로 실행
python src/benchmark.py

# 동시성 10으로 실행
python src/benchmark.py --concurrency 10

# 특정 데이터셋으로 실행
python src/benchmark.py --dataset data/my_dataset.jsonl

# 100개 요청, 동시성 5
python src/benchmark.py --num-requests 100 --concurrency 5
```

## 7. 결과 확인

벤치마크가 완료되면 `results/` 디렉토리에 결과가 저장됩니다:

```bash
results/
├── benchmark_20240206_143022.json  # 상세 결과 (JSON)
├── benchmark_20240206_143022.csv   # 상세 결과 (CSV)
└── benchmark_20240206_143022_plot.png  # 시각화 차트
```

### 결과 분석

JSON 결과 파일 내용:
```json
{
  "total_requests": 100,
  "successful_requests": 98,
  "failed_requests": 2,
  "metrics": {
    "mean_latency": 0.245,
    "p50_latency": 0.230,
    "p95_latency": 0.412,
    "p99_latency": 0.523,
    "requests_per_sec": 12.5,
    "tokens_per_sec": 256.3
  },
  "adapter_stats": {
    "adapter1": {
      "count": 25,
      "avg_latency": 0.240
    },
    ...
  }
}
```

## 8. 고급 사용법

### 커스텀 설정 파일

```bash
# 커스텀 설정 파일 생성
cp config.yaml custom_config.yaml

# 설정 수정 후 실행
python src/benchmark.py --config custom_config.yaml
```

### 특정 어댑터만 사용

`config.yaml`에서 `random_adapter_assignment: false`로 설정하고,
데이터셋에 어댑터를 명시적으로 지정:

```json
{"prompt": "질문", "adapter_name": "adapter1"}
{"prompt": "다른 질문", "adapter_name": "adapter2"}
```

### 대규모 부하 테스트

```bash
# 1000개 요청, 동시성 50
python src/benchmark.py --num-requests 1000 --concurrency 50
```

## 문제 해결

### 서버 연결 실패
- vLLM 서버가 실행 중인지 확인
- 포트가 올바른지 확인 (`config.yaml`의 port 설정)

### GPU 메모리 부족
- `config.yaml`에서 `gpu_memory_utilization` 값을 낮춤 (예: 0.7)
- `max_model_len`을 줄임 (예: 1024)
- `max_loras`를 줄임

### LoRA 어댑터 로드 실패
- 어댑터 디렉토리 경로가 올바른지 확인
- `adapter_config.json` 파일이 존재하는지 확인
- 어댑터와 베이스 모델이 호환되는지 확인

## 다음 단계

- 실제 데이터셋으로 테스트
- 여러 LoRA 어댑터 추가
- 다양한 동시성 레벨 실험
- 결과 분석 및 최적화
