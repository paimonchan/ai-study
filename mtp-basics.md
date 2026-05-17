# MTP (Multi-Token Prediction) — Dasar

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

Hidden state sudah ada sejak awal, bukan fitur MTP. Sebelum MTP, hidden state tidak dipakai setelah lewat `lm_head`.

### Hidden state vs Output

| | Hidden state | Output |
|---|---|---|
| Bentuk | Vektor float (2048 angka) | Vektor probabilitas (248320 angka) |
| Isi | "Pemikiran abstrak" | "Prediksi token selanjutnya" |
| Bisa dibaca? | Tidak — angka acak | Ya — "def 75%, class 10%" |
| Dipakai MTP? | Ya | Tidak |

### Proses dari hidden state ke output

```
Layer N → hidden state [0.9, -0.8, ...]
              ↓
         lm_head — matriks transformasi (2048 × 248320)
              ↓
         [0.75, 0.10, 0.08, ...] ← probabilitas setiap token
```

`lm_head` adalah "penerjemah" yang mengubah hidden state jadi probabilitas.

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
Step 1: target(40 layer) → token A  ← mahal
Step 2: target(40 layer) → token B  ← mahal
Step 3: target(40 layer) → token C  ← mahal

3 token = 3x forward pass target
```

### Dengan MTP
```
Step 1: target(40 layer) → token A + hidden state
Step 2: MTP head(1 layer) → draft [B, C]  ← murah
Step 3: target verify [B, C] ✅✅ → dapat 3 token

3 token = 1x forward pass target + 1x MTP murah
```

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

## Ringkasan Efisiensi

| | Forward pass target | Token per step | Layer ekuivalen |
|---|---|---|---|
| Tanpa MTP | 3× | 3 | 120 |
| Dengan MTP | 1× | 3 | ~41 |

MTP tidak mengurangi jumlah **token output**, tapi mengurangi jumlah **forward pass model besar** dari N kali jadi ~N/3 kali.

---

## Alur MTP di Kode

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
