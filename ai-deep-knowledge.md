# AI Deep Knowledge Base — Seluk Beluk Ekosistem

## Tujuan
Referensi lengkap untuk memahami setiap aspek ekosistem AI modern — dari model, infrastruktur, riset, sampai industri. Bukan cuma checklist harian, tapi pengetahuan yang bisa dirujuk kapan saja.

---

# 📚 BAGIAN 1: MODEL & ARSITEKTUR

## 1.1 Arsitektur Model

### Transformer
| Konsep | Penjelasan | Visual |
|---|---|---|
| **Attention** | Setiap token "memperhatikan" semua token lain. O(n²) — bottleneck utama. | `Q × K^T × V` |
| **Self-attention** | Token dalam satu sequence saling attention | |
| **Cross-attention** | Token dari sequence A attention ke sequence B | |
| **Multi-head Attention** | Banyak head attention paralel, masing-masing fokus ke aspek beda | |
| **Grouped Query Attention (GQA)** | Beberapa query share 1 KV pair. Hemat memori. | 8Q : 4K : 4V |
| **Multi-query Attention (MQA)** | Semua query share 1 KV pair. Paling hemat. | 8Q : 1K : 1V |
| **Flash Attention** | IO-aware attention — GPU SRAM lebih cepat dari HBM | Tiling + recomputation |
| **Sliding Window Attention** | Hanya attention ke N token terdekat — O(n) | Window=4096 |
| **Sparse Attention** | Hanya attention ke subset token yang relevan | Content-dependent |
| **Linear Attention** | Ganti softmax dengan kernel — O(n) ARSITEKTUR | RWKV, Mamba |
| **State Space Model (SSM)** | Ganti attention dengan state recurrent | Mamba, Mamba-2 |
| **MoE (Mixture of Experts)** | Banyak "expert" kecil, router pilih beberapa saja | 256 experts, top-8 |

### MoE Deep Dive
| Aspek | Detail |
|---|---|
| **Router** | Layer kecil yang menentukan expert mana yang aktif |
| **Load balancing** | Pastikan semua expert dipakai rata — loss auxiliary |
| **Expert parallelism** | Setiap expert di GPU beda — scaling horizontal |
| **Top-k routing** | Pilih k expert dengan score tertinggi |
| **Shared expert** | 1 expert yang SELALU aktif — stabilisasi |
| **Active vs Total** | MoE 35B/3B = 35B total, 3B aktif per token |
| **MoE++ (Zyphra)** | MoE dengan CCA — cross-attention untuk koordinasi expert |

### Hybrid Architectures
| Model | Kombinasi |
|---|---|
| **Qwen3.6 35B** | Gated DeltaNet (linear attention) + Gated Attention (full) + MoE — 10 siklus |
| **Mamba-2 + Attention** | SSM untuk sebagian layer, attention untuk sisanya |
| **Nemotron** | Attention hybrid + MoE |
| **ZAYA1 (MoE++)** | CCA block + MoE routing + Mamba-like state |

## 1.2 Quantization

### Levels (dari terbaik ke terendah)
| Quant | Bits per weight | Kualitas | Ukuran relatif |
|---|---|---|---|
| BF16 | 16 | Reference | 100% |
| Q8_0 | 8 | Near-lossless | ~50% |
| Q6_K | 6 | Sangat baik | ~40% |
| Q5_K_M | 5 | Baik | ~35% |
| Q4_K_M | 4 | Cukup | ~28% |
| Q3_K_M | 3 | Lumayan | ~22% |
| Q2_K | 2 | Buruk | ~16% |
| IQ2_XXS | 2.06 | Sangat buruk | ~14% |
| IQ1_M | 1.93 | Eksperimental | ~12% |

### Jenis Quant
| Metode | Cara Kerja |
|---|---|
| **K-quant** (Q4_K_M) | Block-wise, scale per block. Standar llama.cpp |
| **IQ** (Importance-aware) | Bobot penting di-preserve, yang kurang penting di-quant lebih agresif |
| **Unsloth Dynamic 2.0 (UD)** | Per-layer: layer sensitif (attention) di-high precision, FFN di-low precision |
| **HPC** (CompressedGemma) | Beam search + belief propagation — anisotropic error optimization |
| **NVFP4** | Blackwell native FP4 — dukungan hardware langsung |
| **MXFP4** | Microscaling FP4 — format universal |
| **imatrix** | Importance matrix — calibrasi per-tensor, improve kualitas quant |
| **MagicQuant v2** | Hybrid mixed quantization — pipeline otomatis | 

### Aturan Praktis
```
14B dense Q4 → ~9 GB → cocok 12GB VRAM ✅
35B MoE IQ2_M → ~11 GB → cocok 12GB VRAM ✅
80B MoE Q4 → ~48 GB → butuh 48GB+ ❌
27B dense IQ2_XXS → garbage output ❌ (terlalu agresif untuk dense)
```

## 1.3 Tokenizer & Vocabulary
| Model | Vocab size | Approach |
|---|---|---|
| Qwen3.6 | 248.320 | BPE |
| Gemma 4 | 262.144 | SentencePiece |
| Llama 3 | 128.256 | BPE |
| Needle 26M | 8.192 | SentencePiece BPE — sengaja kecil |
| Phi-4 | 100.352 | BPE |

---

# 🚀 BAGIAN 2: SPEED & OPTIMASI

## 2.1 Speculative Decoding

### Perbandingan Metode
| Metode | Draft source | Speedup | Status di llama.cpp |
|---|---|---|---|
| **MTP** | Head built-in (1 layer) | 1.5-2.5x | ✅ Merged (b9180) |
| **Draft model** | Model terpisah (kecil) | 1.5-3x | ✅ `--spec-type draft-simple` |
| **Self-speculative** | History / n-gram | 1.1-1.4x | ✅ `--spec-type ngram-*` |
| **EAGLE** | Hidden state + extra head | 2-3x | ❌ Belum (vLLM: ✅) |
| **DFlash** | Diffusion block (parallel) | 2-3x | ❌ Belum (TPU: ✅) |
| **Orthrus** | Diffusion view (lossless) | up to 7.8x | ❌ Paper 12 Mei 2026 |

### MTP Implementation (kita)
| Parameter | Nilai kita | Optimal menurut Unsloth |
|---|---|---|
| `--spec-type` | `draft-mtp` | `draft-mtp` |
| `--spec-draft-n-max` | 2 | **6** |
| Accept rate | 86% | 70-80% |
| Speed | 48 t/s | 60-80 t/s (target) |

### Draft Model Ecosystem
| Draft model | Ukuran | Untuk |
|---|---|---|
| Qwen3.5-0.8B | ~1.6 GB Q4 | Qwen family |
| Gemma 4 E2B draft | ~2 GB | Gemma 4 (butuh fork) |
| Self-spec (n-gram) | 0 MB | Zero overhead |

## 2.2 KV Cache
| Tipe | Bits | Compression | Kualitas |
|---|---|---|---|
| f16 | 16 | 1x | Reference |
| q8_0 | 8 | 2x | Hampir lossless |
| q4_0 | 4 | 4x | Sangat baik (setelah rotation) |
| TurboQuant (planar3) | 3 | 6x | Lossless (Google, ICLR 2026) |
| PolarQuant | 3.5 | 4.5x | Near-lossless |

### KV Cache Memory Formula
```
KV_cache_memory = 2 × layers × hidden_dim × context_len × bits_per_elem / 8

Contoh Qwen 35B MoE:
  n_layer = 10 (attention layers)
  n_embd_head_k = 256
  n_head_kv = 4
  hidden_dim = 10 × 256 × 4 = 10240  (wait, MoE beda)
  
  Sebenarnya utk Qwen3.6-35B-A3B:
  Hanya attention layer yang punya KV cache (10 layer)
  KV dim = 256, KV heads = 2
  KV_cache_per_token = 10 × 2 × 256 × 2 = 10,240 elemen
  Di q4_0: 10,240 × 4bit = 40,960 bit = 5,120 byte per token
  128K context: 5,120 × 131,072 = ~640 MB
```

## 2.3 Prompt Processing (Prefill)
| Teknik | Speedup | Keterangan |
|---|---|---|
| Flash Attention | 2-5x | IO-aware, fused kernel |
| Chunked prefill | 1.5x | Process prompt dalam chunk |
| Multi-token prediction | 1x | MTP hanya membantu decode, bukan prefill |
| Prompt caching | 1-5x | Cache hasil prefill untuk reuse |

---

# 🔬 BAGIAN 3: RISET & PAPER

## 3.1 Paper Penting yang Perlu Di-track

### KV Cache Compression
| Paper | Tahun | Inti |
|---|---|---|
| TurboQuant | ICLR 2026 | WHT rotation + optimal scalar quantization, 6x compression |
| KIVI | 2024 | Per-channel KV cache quantization |
| KVQuant | 2024 | Per-channel + per-token quantization |
| PolarQuant | 2025 | Geometric compression pada hypersphere |

### Speculative Decoding
| Paper | Tahun | Inti |
|---|---|---|
| Medusa | 2024 | Multiple draft heads parallel |
| EAGLE | 2024 | Feature-level speculation |
| EAGLE-2/EAGLE-3 | 2025 | Improved draft model |
| DFlash | 2025 | Diffusion-style block speculation |
| Orthrus | 2026 | Diffusion view on frozen autoregressive model, up to 7.8x |
| Slow-Fast Inference (SFI) | 2026 | Sparse attention reuse, 1.6-14x |

### MoE
| Paper | Tahun | Inti |
|---|---|---|
| Mixtral | 2024 | Sparse MoE untuk decoder |
| DeepSeek-V2 | 2024 | MLA + MoE |
| Qwen3.6 | 2026 | Gated DeltaNet + MoE hybrid |
| MoE++ (Zyphra) | 2026 | MoE dengan CCA cross-attention |

### New Architectures
| Paper | Tahun | Inti |
|---|---|---|
| Mamba | 2024 | SSM — O(n) inference |
| Mamba-2 | 2024 | Simplified SSM |
| Gated DeltaNet | 2025 | Linear attention + gating — dipakai Qwen3.6 |
| Simple Attention Network (SAN) | 2026 | No FFN — cuma attention + gating (Needle) |

## 3.2 Konferensi

| Konferensi | Jadwal | Relevansi |
|---|---|---|
| **ICLR 2026** | Apr/Mei 2026 | ✅ TurboQuant di-presentasikan |
| **ICML 2026** | Juli 2026 | Quantization, efficiency |
| **NeurIPS 2026** | Des 2026 | Semua topik |
| **CVPR 2026** | Juni 2026 | Vision + multimodal |
| **EMNLP 2026** | Nov 2026 | NLP, LLM, evaluation |

---

# 🏢 BAGIAN 4: INDUSTRI & EKOSISTEM

## 4.1 Perusahaan & Model

| Perusahaan | Model Flagship | Model Open | Strategi |
|---|---|---|---|
| **OpenAI** | GPT-5.5 | ❌ | Proprietary frontier |
| **Anthropic** | Claude Opus 4.6 | ❌ | Safety-first |
| **Google DeepMind** | Gemini 3.1 Pro | ✅ Gemma 4 (Apache 2.0) | Dual strategy: closed frontier + open family |
| **Meta** | Llama 4 | ✅ Llama 4 Scout/Maverick | Open-weight (custom license) |
| **Mistral** | Mistral Large 4 | ✅ Mistral Small 3.2 | Open-weight + commercial |
| **DeepSeek** | DeepSeek V4 | ✅ DeepSeek V3/V4 | MIT license, sangat murah |
| **Alibaba (Qwen)** | Qwen3.6 | ✅ Qwen3.6 family | Apache 2.0, salah satu yang paling open |
| **Zyphra** | ZAYA1-8B | ✅ ZAYA1-8B | MoE++, AMD-native training |
| **Microsoft** | Phi-4 family | ✅ Phi-4 (MIT) | Small models for edge |
| **xAI** | Grok 4.3 | ❌ | Proprietary |
| **Tencent** | Hy3 | ✅ Hy3-preview | MoE 295B |
| **MoonshotAI** | Kimi K2.6 | ✅ K2 base (Apache 2.0) | Agentic coding specialist |
| **Cactus Compute** | Needle 26M | ✅ Needle (MIT) | Tool calling specialist |

## 4.2 Inference Engines

| Engine | Bahasa | Fokus | MTP? | TurboQuant? |
|---|---|---|---|---|
| **llama.cpp** | C/C++ | Local inference | ✅ (draft-mtp) | ❌ (butuh fork) |
| **vLLM** | Python | High-throughput serving | ✅ Gemma4 (v0.21) | ✅ |
| **Ollama** | Go | Easy local use | ❌ | ❌ |
| **LM Studio** | C++ | GUI local | ❌ | ❌ |
| **SGLang** | Python | Structured generation | ❌ | ❌ |
| **KTransformers** | Python | MoE optimization | ❌ | ❌ |
| **MLX** | C++ | Apple Silicon | ❌ | ❌ |
| **TensorRT-LLM** | C++ | NVIDIA optimized | ❌ | ❌ |

## 4.3 Coding Agents

| Tool | Backend | Model Default | Open Source |
|---|---|---|---|
| **OpenCode** | CLI / TUI | Claude / local API | ✅ |
| **Codex** | CLI | GPT / Claude | ✅ |
| **Aider** | CLI | GPT-4o / local | ✅ |
| **Cline** | VS Code | Claude / local | ✅ |
| **Continue** | VS Code / JetBrains | Local / cloud | ✅ |
| **Kilo Code** | CLI | Multi-provider | ✅ |
| **Factory AI** | Cloud | Kimi K2.6 | ⚠️ Platform |

## 4.4 Hosting & Cloud

| Provider | Free Tier | Model Tersedia | Harga |
|---|---|---|---|
| **Groq** | ✅ | Llama, Qwen, Gemma | Per token |
| **Together AI** | $1 credit | Semua open | Per token |
| **Novita AI** | ✅ | Llama, Qwen, DeepSeek | Murah |
| **Hyperbolic** | ✅ | Semua | Per token |
| **DeepInfra** | ✅ | Semua | Per token |
| **Replicate** | ✅ | Semua | Per token |

---

# 🛠️ BAGIAN 5: TOOLCHAIN & BUILD

## 5.1 Build llama.cpp

### Prerequisites (Windows)
| Komponen | Lokasi | Version |
|---|---|---|
| CUDA Toolkit | `E:\AI\LLM\_work\cuda-toolchain\` | 13.1.115 — JANGAN 13.2! |
| MSVC | `...\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe` | 19.39 |
| Windows SDK | `C:\Program Files (x86)\Windows Kits\10\` | 10.0.22621.0 |
| CMake | `E:\AI\LLM\cmake-portable\cmake-3.31.6-windows-x86_64\bin\cmake.exe` | 3.31.6 |
| Ninja | `D:\DataDev\anaconda3\Library\bin\ninja.exe` | dari anaconda3 |

### Build Command (copy-paste ready)
```powershell
$env:PATH = "E:\AI\LLM\_work\cuda-toolchain\Library\bin;D:\DataDev\anaconda3\Library\bin;D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
$env:INCLUDE = "D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\winrt"
$env:LIB = "D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

& cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -S "E:\AI\LLM\llama.cpp-src" -B "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda"

& ninja -C "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda" llama-server

Copy-Item "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda\bin\*" "E:\AI\LLM\llama.cpp\" -Force
```

### Build Flags
| Flag | Fungsi | Kapan dipakai |
|---|---|---|
| `-DGGML_CUDA=ON` | Enable CUDA backend | Selalu (kita punya NVIDIA) |
| `-DGGML_CUDA=OFF` | CPU-only | Testing / debug |
| `-DLLAMA_CURL=OFF` | Disable download support | Sudah off (kita pakai aria2) |
| `-DCMAKE_CUDA_ARCHITECTURES=120` | Blackwell CUDA arch | RTX 5070 adalah Blackwell |
| `-DGGML_CUDA_FA=ON` | Flash Attention CUDA | Default sudah ON |

## 5.2 CMake Generators
| Generator | Cocok? | Alasan |
|---|---|---|
| **Ninja** | ✅ | Paling cepat, tidak butuh VS CUDA integration |
| Visual Studio 17 2022 | ❌ | CUDA toolchain portable tidak punya VS integration files |
| MSBuild | ❌ | Lambat, kompleks |

## 5.3 CUDA Toolchain Notes
- CUDA 13.1 = stable untuk low-bit inference
- CUDA 13.2 = **BUG** — merusak low-bit quantization output
- Toolchain portable di `E:\AI\LLM\_work\cuda-toolchain\` — tidak ada full CUDA install di system
- DLLs (`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`) harus di PATH

---

# ⚡ BAGIAN 6: PERFORMANCE TUNING

## 6.1 Parameter Server

| Flag | Nilai Kita | Efek |
|---|---|---|
| `-c 131072` | 128K | Context window — lebih besar = lebih lambat + lebih banyak VRAM |
| `-ctk q4_0 -ctv q4_0` | q4_0 | KV cache quant — hemat VRAM ~50% dari q8_0 |
| `-np 1` | 1 | Parallel requests — MTP cuma support 1 |
| `-fitt 3000` | 3000 MB | Reserve VRAM — cegah WDDM timeout |
| `--no-mmap` | ON | Load model ke heap — +3GB RAM tapi stabil |
| `--cache-ram 0` | ON | Matikan prompt cache — cegah RAM growth |
| `--no-warmup` | ON | Skip warmup — cold start = warm start (sama saja) |
| `--spec-type draft-mtp` | MTP | Speculative decoding |
| `--spec-draft-n-max 2` | 2 draft | Jumlah draft token — 6 rekomendasi Unsloth |

## 6.2 WDDM Timeout

### Masalah
Windows punya GPU timeout (TDR = Timeout Detection Recovery). Default 2 detik. Kalau kernel CUDA berjalan >2s tanpa response ke display, driver di-reset → llama-server disconnect.

### Solusi
1. **`-fitt N`** — reserve VRAM N MB untuk display driver
2. **Registry**: naikkan TdrDelay
```powershell
# HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
# TdrDelay = 8 (detik)
# TdrDdiDelay = 8
```
3. **Turunkan context** — prompt lebih pendek = lebih cepat

## 6.3 Parameter yang Tidak Kami Pakai

| Parameter | Fungsi | Kenapa Tidak |
|---|---|---|
| `-sm row / --split-mode row` | Split model across GPU | Cuma 1 GPU |
| `--tensor-split` | Tensor parallelism | Cuma 1 GPU |
| `-ot "exps=CPU"` | Offload MoE experts ke CPU | Semua di GPU (cukup VRAM) |
| `--no-kv-offload` | KV cache di CPU | Mau cepat, KV di GPU |
| `-ts` | Tensor split ratio | Cuma 1 GPU |
| `-lb 1024` | Batch size | Default 2048 sudah optimal |
| `--sampling` | Sampling parameters | Default sudah cukup |
| `--mlock` | Lock model in RAM | `--no-mmap` sudah handle |
| `-ctxcp 1` | Context checkpoint | Hanya untuk Gemma 4 SWA |

## 6.4 Speed Cheatsheet
```
Tanpa --no-mmap, tanpa -fitt: 97 t/s  → tercepat, risiko WDDM
Dengan --no-mmap, tanpa -fitt:  62 t/s  → aman, lebih lambat
Dengan --no-mmap + -fitt 3000:  48 t/s  → paling aman, paling lambat
```

---

# 🔍 BAGIAN 7: EVALUASI & BENCHMARK

## 7.1 Coding Benchmarks

| Benchmark | Apa yang diukur | Format | Skor Qwen3.6-35B |
|---|---|---|---|
| **HumanEval** | Python function generation | 164 soal, pass@1 | ~65% |
| **HumanEval+** | Lebih strict test cases | ~164 soal | — |
| **MBPP** | Python basic programming | 974 soal | — |
| **SWE-bench Verified** | Real-world GitHub issues | 500 soal | ~73% |
| **SWE-bench Pro** | Harder software engineering | — | — |
| **Terminal-Bench 2.0** | Terminal/tool use | Multi-step | ~59% (27B) |
| **LiveCodeBench** | Competitive programming | v6, multi-language | ~65% |
| **Aider Polyglot** | Multi-language code editing | Various | — |
| **Codeforces** | Competitive programming | Rating system | — |
| **BigCodeBench** | Diverse programming tasks | 1,140 tasks | — |

## 7.2 General Benchmarks

| Benchmark | Apa yang diukur | Qwen3.6-35B (official) |
|---|---|---|
| **MMLU** | 57 academic subjects | — |
| **MMLU-Pro** | Harder MMLU | — |
| **GPQA Diamond** | PhD-level science | — |
| **MATH** | Competition math | — |
| **AIME 2025/2026** | American Invitational Math | — |
| **GSM8K** | Grade school math | — |
| **HLE** | Humanity's Last Exam | — | 

## 7.3 Benchmark Cara Baca

| Istilah | Arti |
|---|---|
| **pass@1** | Model sukses dalam 1 percobaan |
| **pass@k** | Model sukses dalam k percobaan (biasanya k=5) |
| **n-shot** | Berapa contoh yang diberikan di prompt |
| **CoT** | Chain-of-Thought — model diajak "berpikir" dulu |
| **0-shot** | Tanpa contoh |
| **5-shot** | 5 contoh diberikan |

---

# 🧩 BAGIAN 8: INTEGRASI & DEPLOYMENT

## 8.1 OpenCode Setup

### Provider Config
```json
"llamacpp-q2": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "llama.cpp Q2 (MTP)",
  "options": { "baseURL": "http://127.0.0.1:8081/v1" },
  "models": {
    "qwen-q2": {
      "name": "Qwen3.6 35B IQ2_M MTP",
      "tools": true,
      "reasoning": false,
      "contextWindow": 131072,
      "maxTokens": 16384
    }
  }
}
```

### Key Parameters
| Parameter | Value | Alasan |
|---|---|---|
| `tools: true` | Enable tool calling | Coding agent butuh tools |
| `reasoning: false` | Matikan thinking | Model pake reasoning sendiri |
| `contextWindow: 131072` | 128K | Sesuai server config |
| `maxTokens: 16384` | 16K output | Cukup untuk jawaban coding |

## 8.2 MCP (Model Context Protocol)

### Yang Kita Pakai
| MCP Server | Fungsi |
|---|---|
| `paimon-mcp-fetch` | Web fetching |
| `context7` | Web search / context |
| `sequential-thinking` | Structured thinking |

### Yang Ingin Dicoba
| MCP Server | Untuk |
|---|---|
| `duckduckgo-mcp-server` | Web search (alternatif) |
| `filesystem` | File operations |
| `github` | GitHub API |

## 8.3 Hybrid Cloud Strategy
```
Local (llama.cpp)          Cloud (API fallback)
  ├── Coding cepat            ├── Complex reasoning
  ├── Privacy-sensitive       ├── Long context (SubQ)
  ├── Daily driver            ├── Multimodal (vision)
  └── No cost per token       └── Models too large for 12GB
```

---

# 📈 BAGIAN 9: HARDWARE & PLATFORM

## 9.1 GPU Landscape (Mei 2026)

| GPU | VRAM | Bandwidth | Harga | Model max |
|---|---|---|---|---|
| **RTX 5070** (kita) | **12 GB** | **672 GB/s** | $549 | 35B MoE IQ2_M |
| RTX 4070 | 12 GB | 504 GB/s | ~$550 | 14B Q4 |
| RTX 5070 Ti | **16 GB** | 896 GB/s | $749 | 35B IQ3 / 22B Q4 |
| RTX 5080 | **24 GB** | 1,120 GB/s | $999 | 70B Q4 |
| RTX 5090 | **32 GB** | 1,792 GB/s | $1,999 | 70B Q6 / MoE besar |
| RTX 3090 (used) | **24 GB** | 936 GB/s | ~$800 | Sama dengan 5080 |
| RTX 4090 | 24 GB | 1,008 GB/s | ~$1,500+ | Sama dengan 5080 |

## 9.2 WDDM & Windows Spesifik

| Issue | Cause | Fix |
|---|---|---|
| GPU driver crash | TDR timeout >2s | Registry TdrDelay=8 |
| DLL not found | CUDA path | Tambah PATH |
| `--no-mmap` | File mapping alternatif | +3GB RAM, stabil |
| `start /B` error | Stdin redirection | `0<NUL` |
| Port in use | TIME_WAIT | `netstat -an` cek port |

---

# 🚨 BAGIAN 10: COMMON MISTAKES & LEARNINGS

## Yang Pernah Salah

| Kesalahan | Dampak | Pelajaran |
|---|---|---|
| CUDA DLL tidak di PATH | `STATUS_DLL_NOT_FOUND` | CUDA toolchain di E:, tambah ke PATH |
| `--spec-type mtp` (arg lama) | Unknown speculative type | Upstream rename ke `draft-mtp` |
| Context terlalu besar (256K) | WDDM timeout / RAM penuh | Turun ke 128K |
| 27B IQ2_XXS | Output garbage | 2.06 bpw terlalu agresif untuk dense |
| Prompt cache aktif | RAM naik terus | `--cache-ram 0` |
| -fitt + --no-mmap sama2 aktif | Speed drop 35-50% | Trade-off stabilitas vs speed |
| VS generator | "No CUDA toolset found" | Harus pakai Ninja |
| HF download via hf_transfer | Lambat | Pakai aria2c |
| --draft-max (arg lama) | Argumen removed | Ganti ke `--spec-draft-n-max` |

## Golden Rules
1. **CUDA 13.1 STABLE** — jangan upgrade ke 13.2
2. **Ninja**, bukan Visual Studio generator
3. **Set PATH, INCLUDE, LIB** sebelum build
4. **aria2c** untuk download GGUF — 16 connections
5. **Catat semua** di `E:\AI\study\`
6. **Test dulu** sebelum ganti default

---

# 📋 BAGIAN 11: REFERENSI CEPAT

## 11.1 Commands

### Run
```cmd
llama-start q2          # Start server 128K q4_0 MTP
llama-stop              # Stop server
llama-restart           # Stop + start ulang
llama-test              # Test speed
llama-tail              # Live log
```

### Download Model
```powershell
& "E:\AI\LLM\tools\aria2\aria2c.exe" -x 16 -s 16 -k 5M --continue=true --check-certificate=false -d <dir> -o <file> <url>
```

### OpenCode
```cmd
opencode                 # Default (llamacpp-q2)
opencode -m llamacpp-q2        # Pake Qwen MTP (default)
# opencode -m ollama/gemma4:26b  # Dulu pake Gemma 4 — model removed
```

## 11.2 Paths

| Komponen | Path |
|---|---|
| Binary | `E:\AI\LLM\llama.cpp\llama-server.exe` |
| Models | `E:\AI\LLM\models-external\qwen36-mtp\` |
| Source | `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\` |
| Build | `E:\AI\LLM\_work\llama.cpp-upstream-build-cuda\` |
| CUDA | `E:\AI\LLM\_work\cuda-toolchain\Library\bin` |
| Study | `E:\AI\study\` |
| OpenCode config | `C:\Users\Paimon\.config\opencode\opencode.json` |
| Scripts | `E:\AI\LLM\scripts\` |
| Cache | `E:\AI\LLM\cache\` |

## 11.3 Diagnostics

```powershell
# VRAM
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader

# RAM
$os = Get-CimInstance Win32_OperatingSystem; "Free: $([math]::Round($os.FreePhysicalMemory/1MB,0)) GB / Total: $([math]::Round($os.TotalVisibleMemorySize/1MB,0)) GB"

# llama-server process
Get-Process -Name "llama-server" | Select-Object Id, @{N="WS(MB)";E={[math]::Round($_.WorkingSet64/1MB,0)}}, StartTime

# llama-server arguments
(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'").CommandLine

# CUDA version
& "E:\AI\LLM\_work\cuda-toolchain\Library\bin\nvcc.exe" --version

# Server health
curl -s http://127.0.0.1:8081/health

# Port check
netstat -an | Select-String ":8081.*LISTENING"
```
