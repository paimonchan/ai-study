# Inference Engines — Perbandingan & MTP Support

> Update: Mei 2026

## Overview

Berbagai inference engine untuk LLM dengan dukungan MTP (Multi-Token Prediction):

| Engine | ⭐ GitHub | Bahasa | Format | MTP | Produksi |
|---|---|---|---|---|---|
| **llama.cpp** | 107k | C/C++ | GGUF | ✅ `--spec-type draft-mtp` | Local #1 |
| **vLLM** | 80k | Python/C++ | SafeTensors | ✅ native MTP | Production #1 |
| **SGLang** | 28k | Python/Rust | SafeTensors | ✅ NEXTN algo | DeepSeek specialist |
| **MLX** | 26k | C++/Python | SafeTensors | ✅ mlx-lm #990 | Apple Silicon |
| **TensorRT-LLM** | 13k | C++/Python | SafeTensors | ✅ via Dynamo | NVIDIA enterprise |
| **MTPLX** | baru | Python/MLX | SafeTensors | ✅ dedicated MTP | Apple Silicon only |

---

## llama.cpp (107k ⭐)

- **Bahasa:** C/C++ — zero dependency, single binary (~8 MB)
- **Format:** GGUF
- **MTP:** ✅ via `--spec-type draft-mtp` (PR #22673, merged 16 Mei 2026)
- **Model support:** Qwen 3.6, Gemma 4
- **Platform:** CPU, GPU (CUDA, Metal, Vulkan) — paling portabel
- **Gunakan saat:** local/desktop, single-user, resource terbatas, Mac
- **Tidak cocok:** multi-user high throughput, production serving

```
llama-server \
  --model Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -ngl 99 -c 8192
```

---

## vLLM (80k ⭐)

- **Bahasa:** Python (88%) + C++/CUDA kernels
- **Format:** SafeTensors — **tidak support MTP + GGUF**
- **MTP:** ✅ native, first-class support
- **Model support:** DeepSeek V3/V4, Qwen 3.6, Gemma 4, MiMo, Step3.5 Flash
- **Platform:** NVIDIA, AMD, Intel, TPU — multi-GPU
- **Gunakan saat:** production multi-user, throughput tinggi
- **Tidak cocok:** single GPU kecil (<24 GB), Mac, GGUF

```
vllm serve Qwen/Qwen3.6-27B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

### GGUF + MTP di vLLM?
**Tidak support.** GGUF di vLLM masih "highly experimental and under-optimized". GGUF loader tidak integrated dengan speculative decoding system. MTP di vLLM butuh model arsitektur asli (HuggingFace SafeTensors). Tidak ada rencana untuk GGUF MTP.

---

## SGLang (28k ⭐)

- **Bahasa:** Python (75%) + Rust + CUDA
- **Format:** SafeTensors
- **MTP:** ✅ via `--speculative-algorithm NEXTN`
- **Model support:** DeepSeek V3/R1 (1.76× speedup), MiMo, EXAONE MoE
- **Platform:** NVIDIA, AMD — multi-GPU, expert parallelism
- **Gunakan saat:** DeepSeek serving, multi-node, throughput tinggi
- **Catatan:** MTP via EAGLE wrapper, butuh draft checkpoint terpisah (DeepSeek-V3-NextN)

```
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --speculative-algorithm NEXTN \
  --speculative-draft-model-path SGLang/DeepSeek-V3-NextN \
  --speculative-num-steps 2 \
  --speculative-num-draft-tokens 3
```

---

## MLX (26k ⭐ framework / 5.2k ⭐ mlx-lm)

- **Bahasa:** C++ (65%) + Python (21%) + CUDA (9%) + Metal
- **Format:** SafeTensors
- **MTP:** ✅ — Qwen 3.5/3.6 (PR #990 merged), Gemma 4 via mlx-vlm (#1112)
- **Platform:** Apple Silicon (Metal) + **NVIDIA GPU (CUDA backend sejak 2025)**
- **Gunakan saat:** Apple Silicon, development Mac + deploy ke NVIDIA
- **Catatan:** MTP di CUDA backend masih baru — performa terbaik di Apple Silicon

```
pip install mlx[cuda12]  # untuk NVIDIA GPU
mlx_lm.generate --model Qwen/Qwen3.6-27B --enable-mtp
```

---

## TensorRT-LLM (13k ⭐)

- **Bahasa:** C++ (40%) + Python (51%) + CUDA
- **Format:** SafeTensors
- **MTP:** ✅ via NVIDIA Dynamo (DeepSeek R1)
- **Platform:** NVIDIA GPU only (enterprise)
- **Gunakan saat:** NVIDIA enterprise, production GPU maksimal
- **Catatan:** Setup kompleks, butuh NVIDIA ecosystem

---

## MTPLX (baru)

- **Bahasa:** Python/MLX — engine khusus MTP
- **Format:** SafeTensors
- **MTP:** ✅ dedicated — native MTP runtime
- **Platform:** Apple Silicon **only** — no CUDA/Linux
- **Performa:** ~2.24× speedup Qwen 3.6-27B di M5 Max
- **Catatan:** Engine spesialis MTP, bukan general inference engine

---

## Ringkasan — Engine Mana?

| Kondisi | Pilihan |
|---|---|
| Local, GGUF, single-user | **llama.cpp** |
| Production multi-user, throughput | **vLLM** |
| DeepSeek V3/R1 skala besar | **SGLang** |
| Apple Silicon | **MLX** atau **MTPLX** |
| NVIDIA enterprise | **TensorRT-LLM** |
| Mac + deploy ke NVIDIA | **MLX** (CUDA backend) |

## Yang TIDAK Support MTP

- **ExLlamaV3** — hanya draft model biasa
- **Hugging Face TGI** — belum ada MTP
- **Ollama** — pake llama.cpp backend, tapi belum expose MTP config
- **DeepSeek-Infer Demo** — tidak implement
- **LMDeploy** — masih under evaluation
