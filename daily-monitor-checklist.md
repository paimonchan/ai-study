# AI Daily Monitoring Checklist — Komprehensif

## Tujuan
Pantau perkembangan AI setiap hari biar tidak ketinggalan informasi penting yang relevan dengan:
- **Setup lokal**: RTX 5070 12GB, Ryzen 7 5700X, 32GB RAM
- **Stack**: llama.cpp, MTP speculative decoding, GGUF
- **Use case**: Coding agentic via OpenCode, tool calling

---

## 🔴 Daily (5-10 menit) — Wajib

### 1. Model Baru (GGUF / llama.cpp)
- [ ] **HF trending llama.cpp**: https://huggingface.co/models?other=llama.cpp&sort=trending
  - Cek halaman 1-2, cari model < 15GB
  - Prioritas: MoE 2-5B active, dense < 14B Q4
  - Catat: nama model, size, quant, arsitektur
- [ ] **HF GGUF trending**: https://huggingface.co/models?library=gguf&sort=trending

### 2. llama.cpp
- [ ] **GitHub releases**: https://github.com/ggml-org/llama.cpp/releases
  - Baca release notes — cari: `MTP`, `spec`, `TurboQuant`, `Gemma`, `GGUF`, `KV cache`
  - Perhatian: breaking changes argumen (`mtp` → `draft-mtp`)
  - Build baru setiap ~1-3 hari
- [ ] **Cek isu/PR baru**: https://github.com/ggml-org/llama.cpp/pulls
  - Filter: `label:performance`, `label:enhancement`
  - Keyword: speculative, quantization, flash attention

### 3. General AI News — Semua Model
- [ ] **TechCrunch AI**: https://techcrunch.com/category/artificial-intelligence/ — headline hari ini
- [ ] **VentureBeat AI**: https://venturebeat.com/category/ai/ — industri & funding
- [ ] **The Verge AI**: https://www.theverge.com/ai-artificial-intelligence — consumer AI
- [ ] **ArsTechnica AI**: https://arstechnica.com/ai/ — teknis & analysis
- [ ] **Google AI Blog**: https://ai.googleblog.com/ — research dari Google
- [ ] **Meta AI Blog**: https://ai.meta.com/blog/ — research dari Meta
- [ ] **Microsoft Research**: https://www.microsoft.com/en-us/research/topic/artificial-intelligence/ — research dari Microsoft

### 4. Community — Signal
- [ ] **r/LocalLLaMA**: https://reddit.com/r/LocalLLaMA
  - Trending posts 24h — cari: MTP, quant, "12GB", "5070", "MoE", GGUF
  - Benchmark posts dari user hardware mirip
- [ ] **r/MachineLearning**: https://reddit.com/r/MachineLearning — general ML discussion
  - Cari: "[D] Research", "[P] Project", "[R] Resource"
- [ ] **Hacker News / Show HN**: 
  - Cari: "Show HN: ... model", "GGUF", "llama.cpp", "inference", "local AI", "AI agent"
- [ ] **Lobsters**: https://lobste.rs/t/ai — AI tag, komunitas dev
- [ ] **Twitter/X**: follow akun kunci (Karpathy, Ggerganov, Unsloth, dll)

---

## 🟡 Weekly (20-30 menit) — Penting

### 5. General AI News Roundup
- [ ] **The Batch by Andrew Ng**: https://www.deeplearning.ai/the-batch/ — newsletter mingguan
- [ ] **Import AI by Jack Clark**: https://importai.substack.com/ — AI industry analysis
- [ ] **TLDR AI**: https://tldr.tech/ai — daily digest (cek arsip minggu ini)
- [ ] **Last Week in AI**: https://lastweekin.ai/ — weekly roundup
- [ ] **AI Index Report**: https://hai.stanford.edu/ai-index — Stanford HAI (cek update)
- [ ] **State of AI**: https://www.stateof.ai/ — annual report (cek rilis baru)

### 6. Model Releases Tracker
- [ ] **LLM Stats**: https://llm-stats.com/llm-updates — last 30 days
- [ ] **WhatLLM**: https://whatllm.org/blog/ — artikel bulanan
- [ ] **HF Daily Papers**: https://huggingface.co/papers — model baru + paper
- [ ] **Arxiv**: cari paper baru dengan keyword:
  - `speculative decoding`, `MTP`, `multi-token prediction`
  - `quantization`, `GGUF`, `GPTQ`, `AWQ`
  - `KV cache compression`, `TurboQuant`, `PolarQuant`
  - `MoE`, `mixture of experts`, `routing`
  - `linear attention`, `state space model`, `Mamba`
- [ ] **Synthetic data**: teknik baru generate training data (distillation, self-play, augmentation)

### 7. Speculative Decoding
- [ ] **MTP updates**: ada improvement baru? (acceptance rate, speed)
- [ ] **EAGLE / EAGLE-3**: support llama.cpp? (vLLM sudah)
- [ ] **DFlash**: diffusion-style speculative decoding — masuk llama.cpp?
- [ ] **Self-speculative**: n-gram / lookup decoding improvement
- [ ] **Orthrus**: diffusion view on frozen auto-regressive model (paper 12 Mei)

### 8. Infrastruktur & Build
- [ ] **CUDA toolkit update**: https://developer.nvidia.com/cuda-downloads
  - Cek versi terbaru (saat ini 13.1)
  - Perhatikan: ada bug CUDA 13.2 yang pengaruhi low-bit inference!
- [ ] **NVIDIA driver update**: Game Ready / Studio driver
  - WDDM stability, TDR fixes
- [ ] **CMake / Ninja / MSVC**: update toolchain
- [ ] **Windows Update**: KB yang pengaruhi GPU performa
- [ ] **Long context techniques**: RoPE scaling, YaRN, positional encoding baru?

### 9. Benchmark Landscape
- [ ] **SWE-bench Verified**: https://www.swebench.com/ — SOTA coding
- [ ] **Terminal-Bench 2.0**: leaderboard model baru
- [ ] **LiveCodeBench**: coding benchmark terkini
- [ ] **Aider Polyglot / LLM Leaderboard**: coding comparison
- [ ] **Artificial Analysis**: https://artificialanalysis.ai — model comparison
- [ ] **Agent eval**: τ²-bench, AgentBench, BFCL tool calling benchmark
- [ ] **Multimodal benchmarks**: vision, speech, audio leaderboard

### 10. Quantization
- [ ] **MagicQuant v2.0**: hybrid mixed GGUF — pipeline baru
- [ ] **Unsloth Dynamic 2.0**: update quantization benchmark
- [ ] **imatrix**: importance matrix calibration — tooling update
- [ ] **NVFP4**: Blackwell native FP4 support di llama.cpp/vLLM
- [ ] **TurboQuant status**: Google Research, ICLR 2026 — official release Q2?
- [ ] **Model merging**: DARE, TIES, model interpolation — merge fine-tune

### 11. Agent & Framework
- [ ] **OpenCode**: https://github.com/anthropics/claude-code — update versi
  - Fitur baru: tool calling, MCP, agent mode
- [ ] **vLLM**: https://github.com/vllm-project/vllm/releases
  - v0.21.0 (15 Mei) — Gemma4 MTP support!
  - Release notes: model baru, performance
- [ ] **Ollama**: update versi — support model baru?
  - Catatan: Gemma 4 di Ollama masih bermasalah
- [ ] **LM Studio**: update — GGUF loader, UI improvements
- [ ] **MCP (Model Context Protocol)**: server baru, tool baru
  - https://github.com/modelcontextprotocol/servers
- [ ] **Structured output**: JSON mode, GBNF grammar, function calling format baru
- [ ] **Embedding models**: untuk RAG, retrieval, semantic search
- [ ] **Vision-Language**: LLaVA, Pixtral, Qwen-VL — multimodal lewat llama.cpp?

### 12. Forks & Custom Build
- [ ] **`ikawrakow/ik_llama.cpp`**: https://github.com/ikawrakow/ik_llama.cpp
  - Commits terbaru — Gemma 4 MTP, TurboQuant
  - Bedanya dengan upstream: `-mtp` vs `--spec-type draft-mtp`
- [ ] **`atomic-llama-cpp-turboquant`**: fork dengan TurboQuant
- [ ] **`am17an/mtp-clean`**: sudah merge ke upstream, tidak perlu lagi

---

## 🔵 Monthly (1-2 jam) — Evaluasi

### 13. Model Evaluation
- [ ] **Download model baru** yang promising dan muat 12GB
- [ ] **Test speed**: bandingkan t/s dengan Qwen 35B baseline (48 t/s)
- [ ] **Test quality**: HumanEval subset (20 soal, ~10 menit)
- [ ] **Test VRAM usage**: nvidia-smi sebelum/sesudah
- [ ] **Test MTP compatibility**: `--spec-type draft-mtp` berfungsi?

### 14. Benchmark Ulang
- [ ] **HumanEval full** (164 soal) — kalau ada perubahan signifikan
- [ ] **Bandingkan**: model baru vs Qwen 35B IQ2_M MTP
- [ ] **Catat hasil**: di `benchmarks/human-eval/results/`

### 15. Infra Maintenance
- [ ] **Cleanup disk**: hapus model GGUF yang tidak dipakai
  - `E:\AI\LLM\models-external\qwen36-mtp\`
- [ ] **Cleanup temp files**: `%TEMP%` — installer leftovers, cache
- [ ] **Rebuild llama.cpp** dari upstream master
  - Pull + rebuild: `ninja llama-server`
  - Copy binary ke `E:\AI\LLM\llama.cpp\`
- [ ] **Cek health**: RAM, VRAM, disk space
  - `nvidia-smi`, `Get-CimInstance Win32_OperatingSystem`
- [ ] **Update study repo**: tambah Q&A baru, update checklist

### 16. Docs & Knowledge
- [ ] **Update README.md**: tabel model, speed, benchmark
- [ ] **Update AGENTS.md**: common misses, critical changes
- [ ] **Update `llama-help.bat`**: model list, argumen
- [ ] **Push all repos**: LLM + study

### 17. Security & Licensing

### 18. AI Safety & Ethics
- [ ] **AI Safety Institute**: https://www.aisi.gov.uk/ — publikasi baru
- [ ] **Anthropic Safety**: https://www.anthropic.com/research — safety research
- [ ] **DeepMind Safety**: https://deepmindsafetyresearch.com/ — safety research
- [ ] **CVE advisories**: cek CVE untuk llama.cpp, CUDA, PyTorch, vLLM
- [ ] **Model license changes**: cek lisensi model baru — Apache 2.0? MIT? Custom?
- [ ] **Red-teaming**: laporan baru tentang model vulnerability

### 19. AI Policy & Regulation
- [ ] **EU AI Act**: implementasi, update regulasi
- [ ] **US Executive Orders**: AI policy, export controls
- [ ] **China AI regulation**: regulasi model open-source
- [ ] **Copyright cases**: kasus hukum AI training data
- [ ] **Export controls**: chip embargo (NVIDIA ke China)

### 20. AI in Specific Domains
- [ ] **Medical AI**: new models for diagnosis, drug discovery
- [ ] **Scientific AI**: AlphaFold, weather prediction, physics simulation
- [ ] **Legal AI**: AI for contract analysis, legal research
- [ ] **Financial AI**: trading models, risk assessment
- [ ] **Climate AI**: climate modeling, energy optimization

### 21. Robotics & Embodied AI
- [ ] **New robotics models**: RT-2, π0, generalist robot models
- [ ] **Sim-to-real**: simulation advances for robotics
- [ ] **Humanoid robots**: Figure, Tesla Optimus, Boston Dynamics updates
- [ ] **Robot foundation models**: models that control robot hardware directly

### 22. AI in Creative Fields
- [ ] **Image generation**: Stable Diffusion, Flux, Midjourney updates
- [ ] **Video generation**: Sora, Veo, SANA-WM updates
- [ ] **Music generation**: new audio models
- [ ] **Game AI**: procedural generation, NPC behavior models
- [ ] **Audio/Speech AI**: Whisper, TTS, voice AI — model baru untuk agent pipeline?

### 23. AI Industry & Business
- [ ] **Funding rounds**: startup funding, acquisitions
- [ ] **IPO news**: AI companies going public
- [ ] **Pricing changes**: API pricing shifts (OpenAI, Anthropic, Google)
- [ ] **New entrants**: companies entering AI space
- [ ] **Open source adoption**: perusahaan yang pindah ke open-source models
- [ ] **Cloud AI spending**: tren belanja AI di AWS/GCP/Azure

### 24. Open Source Ecosystem Health
- [ ] **Hugging Face stats**: total models, downloads trend
- [ ] **GitHub stars**: trending AI repos minggu ini
- [ ] **PyPI downloads**: popularity transformers, diffusers, dll
- [ ] **Docker images**: cek new base images for AI
- [ ] **Dev kits**: NVIDIA Jetson, Arduino AI, Raspberry Pi AI

### 25. AI Education & Learning
- [ ] **New courses**: DeepLearning.ai, Fast.ai, Hugging Face course updates
- [ ] **Tutorials**: new practical guides for LLM deployment
- [ ] **Books**: new AI/ML book releases
- [ ] **Workshops**: upcoming conferences, webinars
- [ ] **Datasets**: new benchmark datasets, open training data

### 26. Hardware & Platform — General
- [ ] **CPU**: AMD/Intel new gen — AVX-512, AMX support
- [ ] **NPU**: neural processing units — Snapdragon X, Apple Neural Engine
- [ ] **Cloud GPU**: new instance types (AWS, GCP, Azure)
- [ ] **Edge AI**: Raspberry Pi AI Kit, NVIDIA Jetson updates
- [ ] **Storage**: NVMe SSD, RAM, motherboard for AI workstation
- [ ] **AI energy**: power consumption per model, efisiensi inference per watt
- [ ] **Cek lisensi model baru**: Apache 2.0? MIT? Llama Community?
- [ ] **Cek advisories**: CVE untuk llama.cpp / CUDA / driver
- [ ] **Cek perubahan lisensi**: model yang sebelumnya open-source berubah?

---

## 📊 Tracker Table — Model Candidates

| Model | Rilis | Arsitektur | Size GGUF | VRAM | MTP? | Coding | Catatan |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (kita) | Apr 2026 | MoE 35B/3B | 11.5 GB (IQ2_M) | ✅ 12GB | ✅ | ~65% HE | **Default** |
| CompressedGemma 26B | May 2026 | MoE 25.8B/4B | 9.5 GB | ✅ 12GB | ❌ (fork) | ~60%? | Butuh fork |
| ZAYA1-8B | May 2026 | MoE++ 8B/760M | ~4 GB | ✅ 12GB | ❌ | ~66% LCB | vLLM fork |
| Phi-4 14B | Dec 2024 | Dense 14.7B | ~9 GB (Q5) | ✅ 12GB | ❌ | 82.6% HE | Context 16K |
| Qwen3-4B Toolcall | Mar 2026 | Dense 4B | 4.3 GB | ✅ 12GB | ❌ | N/A | Tool specialist |
| Needle 26M | May 2026 | SAN 26M | 26 MB | ✅ | ❌ | N/A | Tool router |

---

## 🏗️ Infrastructure Reference

| Komponen | Path / Lokasi | Catatan |
|---|---|---|
| llama.cpp binary | `E:\AI\LLM\llama.cpp\llama-server.exe` | Build b1-0253fb2, upstream master |
| CUDA toolchain | `E:\AI\LLM\_work\cuda-toolchain\Library\bin` | CUDA 13.1 — jangan upgrade ke 13.2! |
| CMake portable | `E:\AI\LLM\cmake-portable\cmake-3.31.6-windows-x86_64\bin\cmake.exe` | |
| Ninja | `D:\DataDev\anaconda3\Library\bin\ninja.exe` | |
| MSVC | `D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe` | |
| Windows SDK | `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\` | |
| Models | `E:\AI\LLM\models-external\qwen36-mtp\` | |
| OpenCode config | `C:\Users\Paimon\.config\opencode\opencode.json` | |
| Study repo | `E:\AI\study\` | `github.com/paimonchan/ai-study` |
| LLM repo | `E:\AI\LLM\` | `github.com/paimonchan/llm-plan` |

---

## ⚡ Quick Commands

```powershell
# === DAILY ===

# 1. HF trending llama.cpp
curl -s "https://huggingface.co/api/models?other=llama.cpp&sort=trending&limit=5" | ConvertFrom-Json | Select-Object -ExpandProperty id

# 2. llama.cpp latest release
$rel = curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" | ConvertFrom-Json; $rel.tag_name, $rel.published_at

# 3. Cek binary kita lawas?
Get-Item "E:\AI\LLM\llama.cpp\llama-server.exe" | Select-Object Length, LastWriteTime

# === WEEKLY ===

# 4. Cek CUDA version
& "E:\AI\LLM\_work\cuda-toolchain\Library\bin\nvcc.exe" --version

# 5. Cek VRAM + RAM
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
$os = Get-CimInstance Win32_OperatingSystem; "RAM Free: $([math]::Round($os.FreePhysicalMemory/1MB,0)) GB"

# 6. Cek fork ikawrakow latest commit
$fork = curl -s "https://api.github.com/repos/ikawrakow/ik_llama.cpp/commits?per_page=1" | ConvertFrom-Json; $fork[0].sha.Substring(0,7), $fork[0].commit.author.name, $fork[0].commit.message.Split("`n")[0]

# 7. Cek server status
curl -s http://127.0.0.1:8081/health

# === MONTHLY ===

# 8. Disk usage by model dir
Get-ChildItem "E:\AI\LLM\models-external" -Recurse -Filter "*.gguf" | measure -Property Length -Sum | Select @{N="Total GGUF Size(GB)";E={[math]::Round($_.Sum/1GB,2)}}

# 9. Temp cleanup candidate
Get-ChildItem "$env:TEMP" -Recurse -ErrorAction SilentlyContinue | measure -Property Length -Sum | Select @{N="Temp Size(MB)";E={[math]::Round($_.Sum/1MB,2)}}
```

---

## 🚨 Red Flags — Jika terjadi, segera cek

| Tanda | Kemungkinan | Action |
|---|---|---|
| Server disconnect mid-use | WDDM timeout / VRAM OOM | Turunkan context / fit |
| Output garbage | Quant terlalu agresif / CUDA bug | Naikkan quant / cek CUDA version |
| `--spec-type mtp` error | Argumen renamed | Ganti ke `draft-mtp` |
| Build gagal CUDA | Toolchain mismatch | Cek CUDA_PATH, cmake flags |
| Binary tidak jalan | DLL not found | Tambah CUDA bin ke PATH |
| Speed turun drastis | Context terlalu besar / --no-mmap? | Turunkan context / cek flags |
| RAM naik terus | Prompt cache aktif | Tambah `--cache-ram 0` |

---

## Catatan Penting

- **Jangan upgrade CUDA ke 13.2** — ada bug yang merusak low-bit inference
- **Setiap rebuild**: cek release notes untuk argumen baru/deprecated
- **Prioritas**: model llama.cpp compatible > GGUF > muat 12GB
- **Catat semua** di `E:\AI\study\` — jangan andalkan ingatan
- **Daily skip** kalau tidak ada waktu 5 menit pun — yang penting weekly
