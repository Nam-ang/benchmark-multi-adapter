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

#### 5.1 Multi-Adapter 모드 (각 task별로 다른 adapter)

```bash
# 기본 multi-adapter 벤치마크
python src/benchmark.py --mode multi

# 동시성 테스트 (10개 동시 요청)
python src/benchmark.py --mode multi --concurrency 10

# 특정 데이터셋으로 실행
python src/benchmark.py --mode multi --dataset data/my_dataset.jsonl
```

#### 5.2 Baseline 모드 (모든 task에 single multi-task adapter 사용)

```bash
# baseline 벤치마크 (단일 multi-task adapter 사용)
python src/benchmark.py --mode baseline

# 동시성 테스트
python src/benchmark.py --mode baseline --concurrency 10
```

#### 5.3 비교 모드 (baseline과 multi-adapter 성능 비교)

```bash
# baseline과 multi-adapter를 모두 실행하고 비교
python src/benchmark.py --mode compare

# 동시성 테스트로 비교
python src/benchmark.py --mode compare --concurrency 10

# 커스텀 설정으로 비교
python src/benchmark.py --mode compare --config custom_config.yaml
```

비교 모드에서는 다음 결과를 생성합니다:
- 각 모드별 성능 메트릭
- 성능 향상률 (%)
- 비교 차트 (latency, throughput, success rate)
- `results/comparison_{timestamp}.json`: 비교 결과 JSON
- `results/comparison_{timestamp}_plot.png`: 비교 시각화

## Benchmark Modes

### Multi-Adapter Mode (기본값)
각 요청마다 다른 task-specific adapter를 사용합니다. 각 task에 특화된 adapter를 사용하므로 품질은 높지만 adapter 로딩/스위칭 오버헤드가 발생할 수 있습니다.

### Baseline Mode
모든 요청에 하나의 multi-task adapter를 사용합니다. 모든 task를 하나의 adapter로 처리하므로 adapter 스위칭 오버헤드가 없지만, 각 task에 대한 특화 성능은 떨어질 수 있습니다.

설정 파일(`config.yaml`)에서 `baseline_adapter` 항목으로 사용할 multi-task adapter 이름을 지정할 수 있습니다:

```yaml
benchmark:
  baseline_adapter: "multi_task_adapter"  # baseline 모드에서 사용할 adapter
```

### Compare Mode
Baseline과 Multi-Adapter 모드를 모두 실행하고 결과를 비교합니다. 다음 메트릭들을 비교 분석합니다:
- Latency 비교 (Mean, P50, P95, P99)
- Throughput 비교 (Requests/sec, Tokens/sec)
- Success rate 비교
- Adapter별 사용 통계 (Multi-Adapter 모드)

## Benchmark Metrics

- **Latency**: 요청당 응답 시간 (mean, p50, p95, p99)
- **Throughput**: 초당 처리 요청 수
- **Token Throughput**: 초당 생성 토큰 수
- **Success Rate**: 성공한 요청의 비율
- **Adapter Statistics**: 각 adapter별 사용 횟수 및 평균 latency

## Results

벤치마크 결과는 `results/` 디렉토리에 저장됩니다:

### Single Mode (baseline 또는 multi)
- `results/benchmark_{mode}_{timestamp}.json`: 상세 결과
- `results/benchmark_{timestamp}.csv`: CSV 형식 결과
- `results/benchmark_{timestamp}_plot.png`: 시각화 차트

### Compare Mode
- `results/comparison_{timestamp}.json`: 양 모드의 전체 비교 결과
- `results/comparison_{timestamp}_plot.png`: 비교 시각화 (6개의 subplot)
  - Latency 비교 (bar chart)
  - Throughput 비교 (bar chart)
  - Success rate 비교 (bar chart)
  - Latency 개선율 (horizontal bar chart)
  - Adapter별 요청 분포 (bar chart, multi-adapter만)
  - Summary statistics (table)

## Example Comparison Output

```
==============================================================
COMPARISON RESULTS
==============================================================

--- Latency Comparison ---
Metric               Baseline        Multi-Adapter   Improvement
-----------------------------------------------------------------
Mean Latency         1.234s          1.156s          +6.32%
P50 Latency          1.200s          1.120s          +6.67%
P95 Latency          1.456s          1.389s          +4.60%
P99 Latency          1.678s          1.598s          +4.77%
Requests/sec         8.10            8.65            +6.79%
Tokens/sec           4150.23         4430.15         +6.74%

--- Request Statistics ---
Metric                         Baseline        Multi-Adapter
------------------------------------------------------------
Total Requests                 100             100
Successful Requests            98              98
Failed Requests                2               2
Total Duration                 12.35s          11.56s
```
