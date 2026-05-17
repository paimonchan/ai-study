# AI Daily Monitoring Checklist

## Tujuan
Pantau perkembangan AI setiap hari (5-10 menit) biar tidak ketinggalan model baru, tools update, atau teknik optimasi yang relevan dengan setup RTX 5070 12GB.

---

## Daily (5-10 menit)

### 1. Model Baru — Support GGUF / llama.cpp
- [ ] **HuggingFace trending llama.cpp**: https://huggingface.co/models?other=llama.cpp&sort=trending
  - Filter: `other=llama.cpp` → cek model baru di halaman 1-2
  - Cari: MoE model dengan active params 2-4B
  - Skip: dense >10B, MoE total >100B
  
### 2. llama.cpp Releases
- [ ] **GitHub releases**: https://github.com/ggml-org/llama.cpp/releases
  - Cek tag terbaru (b9189, b9190, ...)
  - Baca release notes — cari: MTP, TurboQuant, spec, GGUF
  - Build baru setiap ~1-3x/minggu

### 3. Community Pulse
- [ ] **r/LocalLLaMA** (Reddit): trending posts 24h
  - Cari: MTP, GGUF, quant, "12GB", "5070", optimization
- [ ] **Hacker News** / Show HN: model/tool baru
  - Cari: "Show HN: ... model", "GGUF", "llama.cpp"

---

## Weekly (15-20 menit)

### 4. Model Releases Tracker
- [ ] **LLM Stats**: https://llm-stats.com/llm-updates
  - Cek "Last 30 days" untuk open source releases
- [ ] **WhatLLM**: https://whatllm.org/blog/new-ai-models-may-2026
  - Blog yang rekapitulasi bulanan

### 5. HF Trending Models
- [ ] **GGUF trending**: https://huggingface.co/models?library=gguf&sort=trending
- [ ] **Filter 12GB friendly**:
  - File size < 15GB
  - MoE active < 5B
  - Dense < 14B di Q4

### 6. Paper / Teknik Baru
- [ ] **Arxiv**: speculative decoding, quantization, KV cache compression
  - Keyword: MTP, TurboQuant, speculative decoding, MoE routing

### 7. Fork Update Check
- [ ] **`ikawrakow/ik_llama.cpp`**: cek commits terbaru
  - Gemma 4 MTP, TurboQuant, dll
- [ ] **`ggml-org/llama.cpp`**: pastikan MTP support tidak regresi

---

## Monthly

### 8. Benchmark & Evaluasi
- [ ] **HumanEval** re-run (kalau ada perubahan signifikan)
- [ ] **Bandingkan speed** model kita vs model baru
- [ ] **Update docs**: README.md, plan-*.md, AGENTS.md

### 9. Study Repo Update
- [ ] Tambah Q&A baru ke `E:\AI\study\`
- [ ] Update AGENTS.md dengan insight baru

---

## Quick Command — Cek Semua

```powershell
# 1. HF trending llama.cpp
curl -s "https://huggingface.co/api/models?other=llama.cpp&sort=trending&limit=5" | ConvertFrom-Json

# 2. llama.cpp latest release
curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# 3. Cek binary kita
Get-Item "E:\AI\LLM\llama.cpp\llama-server.exe" | Select-Object Length, LastWriteTime

# 4. Cek upstream (clone terbaru)
# git -C "E:\AI\LLM\_work\llama.cpp-upstream-src" pull
```

---

## Critical Paths Reference

| Komponen | Path |
|---|---|
| llama.cpp source | `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\` |
| Binary | `E:\AI\LLM\llama.cpp\llama-server.exe` |
| CUDA toolchain | `E:\AI\LLM\_work\cuda-toolchain\Library\bin` |
| Models | `E:\AI\LLM\models-external\qwen36-mtp\` |
| Study | `E:\AI\study\` |
| OpenCode config | `$env:USERPROFILE\.config\opencode\opencode.json` |

---

## Catatan Penting

- **Setiap ada model baru**: cek apakah GGUF tersedia, ukuran file, dan apakah support MTP
- **Sebelum rebuild llama.cpp**: cek release notes untuk breaking changes (argumen berubah seperti `mtp` → `draft-mtp`)
- **Kalau nemu model menarik**: catat di `E:\AI\study\` sebelum lupa
- **Prioritas**: yang penting adalah model yang **llama.cpp compatible + GGUF + 12GB VRAM**
