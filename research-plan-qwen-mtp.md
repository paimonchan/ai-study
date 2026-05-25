# Research Plan — Qwen3.6-35B-A3B Quantization & Alternatif

## Tujuan
Menemukan konfigurasi optimal untuk RTX 5070 12GB: balance antara quality, speed, VRAM, dan MTP support.

## Metodologi
1. Mapping semua quant provider untuk model Qwen3.6-35B-A3B
2. Deep dive tiap provider: benchmark, community feedback, file size, speed
3. Filter yang benar-benar cocok 12GB
4. Test: download, run, benchmark HumanEval subset
5. Bandingkan dengan baseline (Unsloth IQ2_M, 48 t/s, 64.6% HE)

---

## Phase 1: Mapping Quant Provider (Current: Unsloth IQ2_M ✓)

| Provider | Metode | Q12GB | MTP? | Status |
|---|---|---|---|---|
| **Unsloth** (kita) | Dynamic 2.0 | IQ2_M (11.5 GB) | ✅ | ✅ Baseline |
| **ByteShape** | ShapeLearn | IQ2_S (~9.8 GB) | ✅ | ❓ Deep dive needed |
| **Thireus** | Dynamic 3.0 | Custom recipe | ✅ | ❓ Deep dive needed |
| **Bartowski** | imatrix | IQ2_M/IQ2_S | Tergantung file | ❓ Deep dive needed |
| **knoopx** | NVFP4 | ? | ? | ❓ Butuh riset |
| **APEX (mudler)** | I-Balanced | Minimal IQ4_XS (19 GB) ❌ | ? | ❌ Tidak muat |

## Phase 2: Deep Dive

### A. ByteShape ShapeLearn
- Metode: Learned optimal datatype per tensor
- File: IQ2_S (2.25 bpw, ~9.8 GB)
- Claim: Quality lebih baik di bitrate yang sama
- Perlu: Benchmark community, perbandingan PPL dengan Unsloth

### B. Thireus Dynamic 3.0
- Metode: Custom per-tensor quant, PPL-optimal
- Bisa generate recipe spesifik untuk 12GB
- Perlu: Test tool suite, generate recipe, download shards

### C. NVFP4 (knoopx)
- Format FP4 native Blackwell
- RTX 5070 support native FP4
- Perlu: llama.cpp build dengan NVFP4 support

### D. Bartowski imatrix
- Metode: Importance matrix calibration
- Wide quant range
- Perlu: Cek IQ2_S/IQ2_M MTP availability

## Phase 3: Test
1. Download candidate GGUF
2. Run dengan config sama (256K, q4_0, MTP)
3. Test speed + VRAM
4. HumanEval subset (20 soal)
5. Bandingkan dengan baseline

## Phase 4: Keputusan
- Pilih yang terbaik: quality + speed + VRAM
- Update config default
- Update docs
