# MTP (Multi-Token Prediction) — Dasar

> **Source code:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\common\speculative.cpp` (baris 380)
> **Server integration:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\tools\server\server-context.cpp` (baris 785-819)
> **Clone:** `github.com/ggml-org/llama.cpp` → `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\`

## Arsitektur MTP

### Apa itu token?
Token = potongan kata yang model "baca". Model tidak paham huruf, cuma paham angka.

```
"hello world" → tokenizer → [15339, 1917] (2 token)
```

### Apa itu forward pass?
Satu kali proses data dari input ke output melalui semua layer.

```
Input → Layer1 → Layer2 → ... → LayerN → output
```

### Apa itu hidden state?
Representasi internal model di suatu layer. Vektor float (misal 2048 angka) yang merupakan "pemikiran abstrak" model.

```
Layer 40 → [0.9, -0.8, 0.2, 0.7, -0.4, ...] ← hidden state
```

Hidden state sudah ada sejak awal (transformer 2017), bukan fitur MTP. Sebelum MTP, hidden state dipakai cuma untuk 1 tebakan token lalu **dibuang** — setiap forward pass menghitung ulang dari awal.

### Hidden state vs Output

| | Hidden state | Output |
|---|---|---|
| Bentuk | Vektor float d_model (misal 2048) | Vektor probabilitas vocab_size (misal 248320) |
| Dimensi | d_model (model-dependent) | vocab_size |
| Isi | "Pemikiran abstrak" — TIDAK berisi token | Probabilitas setiap token di vocabulary |
| Bisa dibaca? | Tidak — angka acak | Ya — "def 75%, class 10%" |
| Dipakai MTP? | Ya | Tidak |

### Proses dari hidden state ke output

```
Layer N → hidden state [0.9, -0.8, ...]   ← 2048 float
              ↓
         lm_head — matriks transformasi (2048 × 248320)
              ↓
         [0.75, 0.10, 0.08, ...]           ← 248320 probabilitas
```

`lm_head` adalah "penerjemah" yang mengubah hidden state jadi probabilitas. lm_head hanya melakukan **dot product** antara hidden state dengan vektor bobot setiap token — tidak ada "pemikiran" di sini.

### Hidden state tidak berisi token
Hidden state = 2048 angka abstrak (representasi terkompresi). BUKAN berisi 2048/248320 calon token. lm_head-lah yang "mendekompresi" jadi daftar probabilitas untuk seluruh vocabulary.

---

## Forward Pass: Prefill vs Decode

### Prefill (pemrosesan prompt awal)
```
Input: [saya, suka, makan, nasi] ← 4 token
Forward pass: 1x (semua token diproses paralel)
Output: logits untuk 4 token sekaligus
Speed: ~900 token/detik
```

### Decode (generasi tiap langkah)
```
Input: [nasi] ← 1 token terakhir
Forward pass: 1x
Output: 1 token baru
Speed: ~20 ms/token (lambat karena cuma 1 token per pass)
```

1 forward pass = semua layer (misal 40 layer) — tidak bisa lompat layer.

### Siapa yang menentukan jumlah langkah?
- **`max_tokens`** — batas dari pengguna
- **EOS token** (`<|im_end|>`) — model sendiri merasa selesai
- **Stop sequence** — dari pengguna
- **Panjang prompt** — harus diproses dulu semua (prefill)

Model tidak pernah tahu berapa langkah yang dibutuhkan. Setiap langkah cuma mikir "1 token ke depan", terus berulang sampai nemu token selesai.

---

## Cara Kerja MTP

### Tanpa MTP
```
Step 1: target(40 layer) → token A  ← mahal, hidden_state dibuang
Step 2: target(40 layer) → token B  ← mahal, hidden_state dibuang
Step 3: target(40 layer) → token C  ← mahal

3 token = 3x forward pass target = 3× mahal
```

### Dengan MTP
```
Step 1: target(40 layer) → token A + hidden_state
Step 2: hidden_state → [MTP block 1 layer] → h_mtp → lm_head_mtp → token B (draft)
Step 3: h_mtp → [MTP block 1 layer] → h_mtp2 → lm_head_mtp2 → token C (draft)
Step 4: target verify [B, C] ✅✅ dalam 1 batch → dapat 3 token

3 token = 1x forward pass target + 2x MTP murah + 1x verify batch
```

MTP masih menggunakan **hidden_state × lm_head**, BUKAN token × lm_head. Bedanya: hidden state diproses dulu lewat transformer block MTP sebelum masuk lm_head MTP.

### MTP = Discount Decode
MTP seperti **diskon langkah**: token pertama bayar penuh (40 layer), token selanjutnya diskon karena cuma 1 layer.

```
Standar: 40 layer/token → mahal
MTP:     40 layer + (1 layer × MTP_depth) → murah
```

### Penting: 1 decode ≠ MTP draft

| Istilah | Arti |
|---|---|
| 1 step | 1 forward pass (bisa target atau MTP) |
| 1 decode | 1 forward pass **target model** (40 layer) → **MTP draft BUKAN decode** |
| 1 forward pass | 1 kali proses data lewat model — target 40 layer, MTP 1 layer |

Jadi kalau MTP menghasilkan 3 token: secara teknis cuma **2 decode** (target: generate + verify) + **2 draft MTP** (bukan decode).

### MTP Draft Tetap Forward Pass
Draft MTP tetap lewat forward pass — tapi modelnya **1 layer**, bukan 40 layer target.

```
Target forward pass: input → L1 → L2 → ... → L40 → hidden_state → token (1 decode)
MTP forward pass:   hidden_state → [MTP block 1 layer] → h_mtp → lm_head_mtp → token (BUKAN decode)
```

Bobot MTP sudah ada di file GGUF yang sama, bukan model terpisah.

### Trade-off: Akurasi vs Murah

MTP draft lebih murah karena hanya 1 layer, tapi **kurang akurat**:

```
MTP 1 layer:    akurasi ~60-70%,   biaya ~3% target
Target 40 layer: akurasi ~95%,     biaya 100%
```

Ini trade-off yang menguntungkan karena:
- Draft tidak perlu sempurna — cukup lebih baik dari random
- Kalau draft salah → verify reject → fallback → **tidak ada output salah**
- Selama accept rate > 0, tetap dapat speedup

### MTP Depth

Beberapa model punya lebih dari 1 MTP head (Qwen3 235B: **14 head**).

```
MTP depth 1: hidden_state → tebak token B
MTP depth 3: hidden_state → tebak token B, C, D
MTP depth 14: hidden_state → tebak token B s/d O
```

Tapi makin dalam, makin tidak akurat:
```
Depth 1 (token ke-2):  akurasi ~70%
Depth 4 (token ke-5):  akurasi ~40%
Depth 9 (token ke-10): akurasi ~20%
```

**Sweet spot: 1-3 head.** Lebih dari itu, draft sering ditolak semua dan malah buang waktu.

### "Gratis Langkah" — Efeknya

Dengan MTP, setelah 1x forward pass mahal, kita mendapat beberapa tebakan tambahan dari MTP head yang ringan — seolah-olah model sudah melakukan langkah ke-2, ke-3, dst tanpa bayar penuh.

**Ibaratnya:**
- Tanpa MTP = beli tiket Rp10.000 setiap naik bus (3× naik = Rp30.000)
- MTP = bayar Rp10.000 sekali, dapat tebakan halte berikutnya dari MTP murah, lalu verify 3 halte barengan

### Verify tetap 40 layer
Verify adalah forward pass target model **full 40 layer juga**. Bedanya: verify berjalan **parallel** — check 2 draft token dalam 1 forward pass.

```
Verify (parallel):
  Input: [A, draft_B, draft_C] → 40 layer → output 3 token sekaligus
```

---

## Arsitektur MTP Head

### MTP = 1 layer tambahan
```
File GGUF:
  ├── Target model: 41 layer MoE (35B total, 3B aktif)
  └── MTP head: 1 layer kecil + lm_head sendiri

Saat run:
  ctx_tgt = 41 layer (target)
  ctx_dft = 1 layer (MTP head, ~3% dari ukuran target)
```

### Kenapa ada 2 lm_head?
Karena hidden state target dan hidden state MTP berbeda secara matematis:

```
Target:   input → layer1 → ... → layer40 → h_target → lm_head_target → output
MTP:      h_target → MTP layer1 → h_mtp → lm_head_MTP → draft
```

`lm_head` target tidak cocok untuk `h_mtp`, dan sebaliknya.

### Kenapa MTP tidak digabung jadi 1 model utuh (42 layer)?
Karena 42 layer = setiap forward pass makin lambat. MTP malah jadi beban. Makanya MTP dipisah jadi model kecil yang **hanya dipanggil pas butuh draft**.

---

> **Source:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\common\speculative.cpp` (baris 380)
> **Server integration:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\tools\server\server-context.cpp` (baris 785-819)

### Layer arsitektur
```
server.cpp → server-context.cpp (load_model, update_slots)
                → common/speculative.cpp (MTP state machine)
                    → src/llama-ext.h (hidden state API)
```

### API MTP
```
common_speculative_init()    — buat speculative state
common_speculative_process() — feed hidden state ke MTP
common_speculative_draft()   — generate draft token
common_speculative_accept()  — verify & accept/reject
```

### Di server-context.cpp (line 802)
```
cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP
ctx_dft = llama_init_from_model(model_tgt, cparams_mtp)
```
