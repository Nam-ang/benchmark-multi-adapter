# vLLM Multi-LoRA Benchmark

vLLM 기반 multi-LoRA 어댑터에 대한 벤치마크 테스트 도구입니다.

## Features

- 여러 LoRA 어댑터를 동시에 로드하여 성능 테스트
- 데이터셋을 랜덤 셔플링하여 각 인스턴스에 무작위 어댑터 할당
- 동시성 부하 테스트 지원
- 상세한 성능 메트릭 수집 (latency, throughput, GPU 메모리 등)

## Installation

```bash
pip install -r requirements.txt
```

## Directory Structure

```
dynamic-lora/
├── data/               # 데이터셋 디렉토리
├── adapters/           # LoRA 어댑터 디렉토리
├── src/
│   ├── server.py      # vLLM 서버
│   ├── benchmark.py   # 벤치마크 클라이언트
│   └── utils.py       # 유틸리티 함수
├── config.yaml        # 설정 파일
└── results/           # 벤치마크 결과
```

## Usage

### 1. 데이터셋 준비

`data/` 디렉토리에 JSONL 형식의 데이터셋을 배치하세요:

```json
{"prompt": "질문 내용", "adapter_name": "adapter1"}
```

### 2. LoRA 어댑터 준비

`adapters/` 디렉토리에 LoRA 어댑터를 배치하세요:

```
adapters/
├── adapter1/
│   └── adapter_model.safetensors
├── adapter2/
│   └── adapter_model.safetensors
└── ...
```

### 3. 설정 파일 수정

`config.yaml`에서 모델, 어댑터, 벤치마크 설정을 조정하세요.

### 4. vLLM 서버 실행

```bash
python src/server.py
```

### 5. 벤치마크 실행

```bash
# 기본 벤치마크
python src/benchmark.py

# 동시성 테스트 (10개 동시 요청)
python src/benchmark.py --concurrency 10

# 특정 데이터셋으로 실행
python src/benchmark.py --dataset data/my_dataset.jsonl

# 커스텀 설정으로 실행
python src/benchmark.py --config custom_config.yaml
```

## Benchmark Metrics

- **Latency**: 요청당 응답 시간 (p50, p95, p99)
- **Throughput**: 초당 처리 요청 수
- **Token Throughput**: 초당 생성 토큰 수
- **GPU Memory**: GPU 메모리 사용량
- **Adapter Switch Time**: 어댑터 전환 시간

## Results

벤치마크 결과는 `results/` 디렉토리에 저장됩니다:
- `results/benchmark_{timestamp}.json`: 상세 결과
- `results/benchmark_{timestamp}.csv`: CSV 형식 결과
- `results/benchmark_{timestamp}_plot.png`: 시각화 차트
