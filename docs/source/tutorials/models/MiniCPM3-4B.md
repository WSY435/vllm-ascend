# MiniCPM3-4B

## Introduction

MiniCPM3-4B is the 3rd generation of the MiniCPM series from OpenBMB. With only 4B parameters, its overall performance surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, and is comparable to many recent 7B~9B models. It supports function call and code interpreter, and has a 32k context window.

Architecturally, MiniCPM3-4B adopts a DeepSeek-V2 style Multi-head Latent Attention (MLA), which compresses the KV cache through low-rank projections (`q_lora_rank`, `kv_lora_rank`) together with decoupled RoPE (`qk_nope_head_dim` + `qk_rope_head_dim`). For MiniCPM3-4B, the attention head dimension `qk_head_dim = 64 + 32 = 96`.

> **Ascend adaptation note**: The Ascend fused infer attention kernel (`npu_fused_infer_attention_score`, TND layout) only supports head sizes in `{64, 128, 192}`. Since `96` is not supported, vllm-ascend ships an adaptation patch (`vllm_ascend/patch/worker/patch_minicpm3.py`) that zero-pads `q/k/v` to the nearest supported head size (`128`) before attention. The padded dims are zeros, so the dot-product scores and output are numerically identical to the upstream implementation.

The `MiniCPM3-4B` model was supported since `vllm-ascend:v0.18.0`.

## Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

| Feature | Support |
| --- | --- |
| Single-card inference | Yes |
| Tensor parallel (multi-card) | Yes |
| BF16 | Yes |
| Prefix caching | Yes |
| Chunked prefill | Yes |

## Environment Preparation

### Model Weight

- `MiniCPM3-4B` (BF16 version): require 1 Atlas 910B3 (64G × 1) card for single-card deployment. [Download model weight](https://huggingface.co/openbmb/MiniCPM3-4B)

It is recommended to download the model weights to a local directory (e.g., `./MiniCPM3-4B/`) for quick access during deployment.

### Installation

You can use our official docker image.

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-openeuler
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -it $IMAGE bash
```

## Deployment

### Single-card Deployment

MiniCPM3-4B supports single-card deployment on the 910B3 platform. Follow these steps to start the inference service:

1. Prepare model weights: Ensure the downloaded model weights are stored in the `./MiniCPM3-4B/` directory.
2. Create and execute the deployment script (save as `deploy.sh`):

```shell
#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH="./MiniCPM3-4B"

vllm serve ${MODEL_PATH} \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name minicpm3-4b \
          --trust-remote-code \
          --max-model-len 32768 \
          --dtype bfloat16
```

### Multi-card (Tensor Parallel) Deployment

MiniCPM3-4B supports tensor parallel inference on multiple cards. For example, on a 2-card node:

```shell
#!/bin/sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MODEL_PATH="./MiniCPM3-4B"

vllm serve ${MODEL_PATH} \
          --host 0.0.0.0 \
          --port 8000 \
          --served-model-name minicpm3-4b \
          --trust-remote-code \
          --max-model-len 32768 \
          --dtype bfloat16 \
          --tensor-parallel-size 2
```

### Prefill-Decode Disaggregation

Not supported yet.

## Functional Verification

After starting the service, verify functionality using a `curl` request:

```shell
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "minicpm3-4b",
        "prompt": "推荐5个北京的景点。",
        "max_completion_tokens": 64,
        "temperature": 0
    }'
```

A valid response (e.g., `"颐和园、故宫、八达岭长城、天坛、北海公园"`) indicates successful deployment.

You can also verify with the offline `LLM` API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="./MiniCPM3-4B",
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=4096,
    tensor_parallel_size=1,
)
sp = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=64)
print(llm.generate(["推荐5个北京的景点。"], sp)[0].outputs[0].text)
```

## Accuracy Evaluation

### Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

A sample accuracy report is shown below:

| dataset | version | metric | mode | vllm-api-general-chat |
|----- | ----- | ----- | ----- |--------------|
| gsm8k | - | accuracy | gen | 81.00 |
| ceval-valid | - | accuracy | gen | 73.00 |

## Performance

### Using vLLM Benchmark

Run performance evaluation of `MiniCPM3-4B` as an example. Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

```shell
vllm bench serve \
  --model ./MiniCPM3-4B/ \
  --dataset-name random \
  --random-input 200 \
  --num-prompts 200 \
  --request-rate 1 \
  --save-result \
  --result-dir ./perf_results/
```

After several minutes, you can get the performance evaluation result.

## Troubleshooting

### `aclnnFusedInferAttentionScoreV3` tiling error with head size 96

If you see an error like `queryD(96), keyD(96) and valueD(96) must be same equal 192/128/64`, it means the Ascend attention kernel rejected the `qk_head_dim = 96` of MiniCPM3. Make sure the `vllm_ascend` plugin is loaded (it is auto-loaded via the `ascend` platform plugin) so that `patch_minicpm3` is applied, which pads the head size to `128`.

### `trust_remote_code` is required

MiniCPM3 ships custom modeling code, so `--trust-remote-code` (or `trust_remote_code=True`) must be passed when loading the model.
