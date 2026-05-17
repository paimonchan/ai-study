# AGENTS.md — Study & Knowledge Rules

## Purpose
This repository is a **living knowledge base** of AI/ML concepts learned through deep-dive Q&A sessions. Every file here represents a topic we explored interactively — question by question, concept by concept.

## How We Learn
1. **Start with a question** — pick a topic, ask a specific question
2. **Deep dive iteratively** — explore layer by layer, question by question
3. **Use analogies** — explain concepts with real-world analogies before diving into math/code
4. **Anchor to code** — after conceptual understanding, link to actual code (llama.cpp source, CUDA kernels, etc.)
5. **Document the journey** — save the Q&A as a structured markdown file for future reference

## Q&A Format

When doing a deep-dive session, structure the output as:

```markdown
## Topic Name

### Question 1: [conceptual question]
[Clear, concise answer with analogy]

### Question 2: [follow-up question]
[Deeper explanation, may reference code]

...
```

### File Naming
- `topic-name.md` — one file per topic
- If a topic grows large, split into `topic-name-part1.md`, `topic-name-part2.md`
- Use lowercase with hyphens

### Content Guidelines
- **Analogies first** — always lead with a real-world analogy before technical details
- **Code references** — link to specific files/lines in llama.cpp or other relevant codebases
- **Visual ASCII diagrams** — use simple text diagrams to illustrate flows
- **Summaries** — include a summary/cheatsheet at the end of each file
- **Progressive complexity** — start simple, get detailed

## Topics Covered So Far

| File | Topics |
|---|---|
| `ai-basics.md` | Token, forward pass, hidden state, lm_head, autoregressive, prefill vs decode, speculative decoding |
| `mtp-basics.md` | MTP architecture, hidden state vs output, verify flow, 2 lm_head, efficiency, draft acceptance |
| `llamacpp-code.md` | server.cpp structure, MTP state machine, build setup, CUDA toolchain, MTP spec code |
| `daily-monitor-checklist.md` | **Daily/weekly/monthly monitoring**: model tracker, llama.cpp releases, community, benchmark landscape, speculative decoding status, quantization, infrastructure, forks, security, red flags, quick commands |
| `ai-deep-knowledge.md` | **11 sections comprehensive reference**: model architectures, MoE deep dive, quantization levels, speculative decoding methods, KV cache, paper tracker, industry players, build guide, performance tuning, WDDM, benchmarks, OpenCode config, hardware landscape, common mistakes, diagnostics |

## Maintenance
- **Update files setiap selesai Q&A session** — review percakapan, tangkap insight baru, perbaiki analogi yang kurang jelas
- Add cross-references between related topics
- Review and refine analogies for clarity
- Pastikan perbedaan dimensi (d_model vs vocab_size, vektor vs token) selalu jelas
- Commit with descriptive messages: `study: add [topic] Q&A` atau `study: update [topic] — klarifikasi [aspect]`

## Quick Reference — Key Concepts

### Local Source Code Paths
| Component | Path |
|---|---|
| llama.cpp source | `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\` |
| llama.cpp build | `E:\AI\LLM\_work\llama.cpp-upstream-build-cuda\` |
| llama-server binary | `E:\AI\LLM\llama.cpp\llama-server.exe` |
| MTP spec code | `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\common\speculative.cpp` (baris 380) |
| Server context | `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\tools\server\server-context.cpp` (baris 785-819) |
| CUDA toolchain | `E:\AI\LLM\_work\cuda-toolchain\Library\bin` |

### Daily Monitoring Quick Command
```powershell
# HF trending + llama.cpp release + binary age
curl -s "https://huggingface.co/api/models?other=llama.cpp&sort=trending&limit=3" | ConvertFrom-Json | % id; $r=curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"|ConvertFrom-Json; $r.tag_name; Get-Item "E:\AI\LLM\llama.cpp\llama-server.exe"|Select Length,LastWriteTime
```

### Red Flags
| Tanda | Action |
|---|---|
| Server disconnect mid-use | WDDM timeout / VRAM OOM → turunkan context / fit |
| Output garbage | Quant terlalu agresif / CUDA 13.2 bug → naikkan quant |
| `--spec-type mtp` error | Argumen rename ke `draft-mtp` |
| Build gagal CUDA | Toolchain mismatch → cek CUDA_PATH, cmake flags |
| Binary tidak jalan (STATUS_DLL_NOT_FOUND) | CUDA bin tidak di PATH → tambah `E:\AI\LLM\_work\cuda-toolchain\Library\bin` |
| RAM naik terus | Prompt cache aktif → `--cache-ram 0` |

### Token
Potongan kata yang model baca sebagai angka. Tokenizer ubah teks → token IDs.

### Forward Pass
Satu kali proses data dari input ke output melalui semua layer neural network.

### Decode (= Step = Forward Pass Target)
1 decode = 1 step = 1 forward pass target model (40 layer). MTP draft BUKAN decode — hanya forward pass 1 layer.

### Hidden State
Vektor float yang merupakan "pemikiran abstrak" model di suatu layer. Bukan probabilitas. Dimensinya d_model (bukan jumlah token).

### lm_head
Matriks proyeksi linear (BUKAN otak) yang memetakan hidden state (d_model) → probabilitas (vocab_size). Bobotnya di-learn saat training, bukan dibuat manual.

### Autoregressive
Generate token satu per satu — setiap langkah cuma nebak 1 token ke depan, terus berulang.

### Speculative Decoding
Teknik percepat inference: model kecil generate draft → model besar verify dalam batch.

### MTP (Multi-Token Prediction)
Variant speculative decoding dengan head built-in di GGUF. Draft pakai 1 layer (bukan target 40 layer) — murah tapi kurang akurat. Trade-off: draft tidak perlu sempurna, cukup lebih baik dari random. MTP depth >3 sering diminishing returns.

### vLLM + MTP + GGUF
vLLM support MTP (SafeTensors) dan GGUF (experimental), tapi **tidak support MTP dengan GGUF**. MTP butuh arsitektur model asli via HuggingFace. GGUF loader tidak integrated dengan speculative decoding. Tidak ada rencana.

### MLX ≠ Apple-only
MLX sekarang punya CUDA backend (NVIDIA GPU, Linux/Windows). Tapi MTP di CUDA masih baru — performa terbaik tetap di Apple Silicon.
