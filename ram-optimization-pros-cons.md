# RAM Optimization — Pros & Cons

Berdasarkan deep dive dari Reddit, GitHub Issues, PR, dan artikel.

---

## 1. `--n-cpu-moe N` — Keep MoE Experts in CPU

> **Untuk model MoE** (Qwen3.6-35B-A3B, Qwen3-Coder-Next, dll)

### Cara kerja
MoE model punya dense layers (attention, dll) dan sparse layers (expert FFN). `--n-cpu-moe N` memindahkan expert weights dari N layer pertama ke CPU. Dense layers tetap di GPU.

### Hasil dari forum (Qwen3-Coder 30B-A3B di 8GB VRAM):

| Config | VRAM used | VRAM free | Speed |
|---|---|---|---|
| Semua di GPU (baseline) | ~7.9 GB | ~0.3 GB | 33.6 t/s |
| `--n-cpu-moe 40` | 7.3 GB | **0.8 GB** | **32.5 t/s** ✅ |
| `--n-cpu-moe 42` | 6.6 GB | 1.5 GB | 30.3 t/s |
| `--n-cpu-moe 44` | 5.9 GB | 2.1 GB | 29.4 t/s |
| `--cpu-moe` (all) | 4.4 GB | 3.6 GB | 13.4 t/s |

### Pros
| Pro | Detail |
|---|---|
| **Hemat VRAM** signifikan | Setiap layer MoE yang dipindahkan free ~400-800 MB VRAM |
| **Dense layers tetap GPU** | Attention, embedding, dll tetap cepat |
| **Speed loss minimal** kalau N pas | 40 layer ≈ 32.5 t/s vs baseline 33.6 t/s (hanya -3%) |
| **Bisa fine-tune** | N kecil = lebih cepat, N besar = lebih hemat VRAM |

### Cons
| Con | Detail |
|---|---|
| **Nambah RAM** | Expert weights di CPU = tambah ~1-4 GB RAM |
| **Butuh tuning** | N tergantung model, VRAM free target, speed target |
| **Multi-GPU complex** | Urusan tensor split + n-cpu-moe barengan ribet |
| **MoE-specific** | Tidak berguna untuk dense model |

### Untuk setup kita (35B MoE, 12GB VRAM)
**Potensi:** Kita punya ~2 GB VRAM free sekarang. Dengan `--n-cpu-moe 38` bisa free ~0.8 GB tambahan. Tapi RAM bakal naik ~1-2 GB — kontraproduktif dengan goal kita (RAM turun).

---

## 2. `--no-mmap` (sekarang) vs `--load-mode mmap`

### State sekarang: `--no-mmap`

| Pro | Con |
|---|---|
| Layer GPU-offload dibebaskan dari RAM ✅ | Model di-heap (private, tidak reclaimable) |
| WS stabil 9.4 GB | Load ulang lambat |
| Tidak ada risiko pageout/swap | — |

### Alternatif: mmap (default)

| Pro | Con |
|---|---|
| OS bisa reclaim page cache ✅ | WS terlihat besar (11.6 GB) — tapi reclaimable |
| Load ulang instant (cached) | GPU-offloaded layers tetap di page cache |
| Speed 97 t/s (tanpa -fitt) | — |

**Verdict:** `--no-mmap` lebih hemat RAM secara nyata untuk kita karena GPU offload layers dibebaskan. Forum Issue #9059 mengkonfirmasi ini. **Stay.**

---

## 3. `--fitt 3000` → `--fitt 1500`

| Pro | Con |
|---|---|
| Free ~1.5 GB RAM | Risiko WDDM timeout lebih tinggi |
| Tanpa -fitt speed 97 t/s | Tapi test sebelumnya aman |

**Verdict:** Bisa turunin ke 1500 atau 0 kalau mau hemat RAM. Risiko WDDM rendah.

---

## 4. `-c 262144` → `-c 131072` (256K → 128K)

| Pro | Con |
|---|---|
| Hemat VRAM ~680 MB | Context terbatas 128K |
| Tidak ngaruh ke RAM (KV di VRAM) | — |

**Verdict:** Efek ke RAM minimal. Hanya pengaruh VRAM.

---

## 5. `--cache-ram` — Spill KV Cache ke RAM

| Pro | Con |
|---|---|
| Bisa running model besar di VRAM terbatas | **Nambah RAM massive** (KV cache 256K = ~2.7 GB) |
| — | Speed turun 30-50% |
| — | Forum bilang: "last resort, avoid if possible" |

**Verdict:** ❌ Jangan. Kontraproduktif.

---

## 6. `--mlock` — Lock Model di RAM

| Pro | Con |
|---|---|
| Model tidak di-swap | **Nambah RAM** (model di-pin) |
| Performa lebih stabil | Tidak berguna kalau tidak ada pressure RAM |

**Verdict:** ❌ Jangan. Kita tidak butuh.

---

## 7. Stop OpenCode Lama (PID 10708)

| Pro | Con |
|---|---|
| Free ~800 MB RAM ✅ | Harus restart session |
| Sudah jalan 18+ jam | — |

**Verdict:** ✅ Langsung lakukan.

---

## 8. Restart llama-server Periodik

| Pro | Con |
|---|---|
| Reset buffer allocations ✅ | Downtime ~30-60 detik |
| Bebasin RAM yang terakumulasi | — |

**Verdict:** ✅ Bisa cron setiap 6-8 jam.

---

## 9. `--n-cpu-moe-draft` — MoE Offload untuk Draft Model

Untuk MTP head. Sama seperti `--n-cpu-moe` tapi untuk draft model (MTP head). Efek minimal karena MTP head cuma 1 layer.

---

## Ringkasan Prioritas untuk Setup Kita

| Prioritas | Aksi | Hemat RAM | Risiko |
|---|---|---|---|
| **1** | Stop OpenCode PID 10708 | **~800 MB** ✅ | None |
| **2** | Turun `-fitt` 3000 → 1500 | **~1.5 GB** | WDDM (rendah) |
| **3** | Restart llama-server periodik | **Buffer reset** | Downtime 30s |
| **4** | `--n-cpu-moe N` (kalau perlu) | **~0.5-2 GB VRAM** | **Nambah RAM** ⚠️ |
| — | Hapus `--no-mmap` | ❌ Nambah RAM (Windows) | Issue #14187 |
| — | `--cache-ram` | ❌ Nambah RAM | — |
| — | `--mlock` | ❌ Nambah RAM | — |

---

## 10. `GGML_CUDA_NO_PINNED=1` — ⭐ Temuan Paling Menjanjikan

> **Sumber:** GitHub Issue #14187 (Windows), PR #1233, PR #6083, Issue #1533 (ikawrakow fork)

### Apa itu pinned memory?
llama.cpp secara default menggunakan pinned memory (CUDA_Host buffer) untuk transfer data CPU↔GPU lebih cepat. Di memory breakdown kita:

```
CUDA_Host model buffer size = ~3238 MiB  ← INI PINNED MEMORY
```

Pinned memory **tidak bisa di-swap atau di-reclaim** — tetap di RAM fisik.

### Efek GGML_CUDA_NO_PINNED=1
Pinned memory dialokasikan via `cudaMallocHost`. Dengan env var `GGML_CUDA_NO_PINNED=1`, llama.cpp fallback ke pageable memory biasa.

| Tanpa env var (default) | Dengan `GGML_CUDA_NO_PINNED=1` |
|---|---|
| CUDA_Host buffer ~3.2 GB | **Tidak dialokasikan** ✅ |
| Transfer CPU→GPU lebih cepat | Transfer lebih lambat (teoritis) |
| Memory tidak bisa di-reclaim | Memory bisa di-swap OS |

### Test:
```powershell
$env:GGML_CUDA_NO_PINNED = "1"
llama-server ...
```

### Dari GitHub Issue #14187 (Windows specific):
> *"When mmap = true, layers allocated to GPU still consume system memory on Windows. --no-mmap fixes this. This does NOT affect Linux."*

**Ini konfirmasi: `--no-mmap` + GPU offload adalah pilihan tepat untuk Windows.**

### Dari Issue #1533 (ikawrakow fork):
> *"GGML_CUDA_NO_PINNED=1 eliminates extra loading time and restores old functionality. Negligible difference for PP compared to older build."*

---

## 11. Windows-Specific Findings

### Issue #14187 — Windows RAM Double Allocation
| Kondisi | RAM |
|---|---|
| `mmap` (default) + GPU offload | **Model di page cache + salinan di RAM** ❌ |
| `--no-mmap` + GPU offload | Layer GPU dibebaskan dari RAM ✅ |

**Ini alasan kenapa `--no-mmap` benar untuk kita.** Windows tidak behave seperti Linux — mmap menyebabkan double allocation.

---

## 12. Kesimpulan Final — Semua Opsi

| Prioritas | Aksi | Hemat RAM | Risiko | Sumber |
|---|---|---|---|---|
| **🟢 1** | Stop OpenCode PID 10708 | **~800 MB** | None | — |
| **🟢 2** | `$env:GGML_CUDA_NO_PINNED=1` | **~3.2 GB** 🔥 | Speed? (perlu test) | Issue #14187, PR #1233 |
| **🟢 3** | Stop/restart llama-server | Reset buffer | Downtime 30s | — |
| **🟡 4** | Turun `-fitt` 3000 → 1500 | **~1.5 GB** | WDDM (rendah) | — |
| ⚪ 5 | `--n-cpu-moe N` | VRAM, **tambah RAM** | Speed turun | PR #15077 |
| ❌ | Hapus `--no-mmap` | ❌ Nambah RAM (Windows) | — | Issue #14187 |
| ❌ | `--cache-ram` / `--mlock` | ❌ Nambah RAM | — | |

### Ringkasan
1. **`--no-mmap` sudah benar** — Windows butuh ini agar GPU-offloaded layers tidak dobel di RAM
2. **Pinned memory (`GGML_CUDA_NO_PINNED=1`)** adalah opsi PALING MENJANJIKAN — bisa hemat ~3.2 GB
3. **Tidak ada magic flag lain** — forum dan GitHub sudah exhausted semua opsi
4. **Kalau mau hemat lebih:** kurangi `-fitt`, restart periodik, stop OpenCode lama
