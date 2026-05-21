# Qwen3.6-35B-A3B — Quantization Options (Research)

## Sumber Quant (GGUF)

| Provider | Metode | Kelebihan | File untuk 12GB |
|---|---|---|---|
| **Unsloth** (kita) | Dynamic 2.0 | Per-tensor dynamic, SOTA KLD | `UD-IQ2_M` (11.5 GB) ✅ |
| **ByteShape** | ShapeLearn | Learned optimal datatype per tensor | `IQ2_S-2.25bpw` (~9.8 GB) ✅ |
| **Bartowski** | imatrix | Wide quant coverage | `IQ2_M` (~11 GB) ✅ |
| **Thireus** | Dynamic 3.0 | Custom recipe per hardware | Perlu generate sendiri |
| **knoopx** | NVFP4 | Native Blackwell FP4 | Perlu cek size |
| **APEX (mudler)** | I-Balanced | Agentic coding, KL max rendah | Minimal IQ4_XS (19.6 GB) ❌ |

## Yang Cocok 12GB VRAM

| File | Provider | bpw | Size | MTP? |
|---|---|---|---|---|
| `UD-IQ2_M` | Unsloth | 2.7 | 11.5 GB | ✅ (kita) |
| `UD-IQ2_XXS` | Unsloth | 2.06 | 10.8 GB | ✅ |
| `IQ2_S-2.25bpw` | **ByteShape** | 2.25 | ~9.8 GB | ✅ ⭐ |
| `IQ2_S` | Bartowski | ~2.5 | ~10 GB | ✅ |
| `IQ2_M` | Bartowski | ~2.7 | ~11 GB | ✅ |
| `NVFP4` | knoopx | ~4 | ~? | ❓ |

## Catatan

- **ByteShape ShapeLearn** — Alternatif paling menarik. Klaim quality lebih baik di 2.25 bpw.
- **Thireus Dynamic 3.0** — Bisa bikin custom quant tailored untuk 12GB VRAM kita. Tapi butuh tool suite.
- **Semua provider** terpengaruh CUDA 13.2 bug — kita pakai 13.1 aman.
- **CUDA 13.2 bug** merusak low-bit inference — fix di 13.3. Kita stay di 13.1.
