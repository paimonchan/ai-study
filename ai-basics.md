# AI Inference — Konsep Dasar

> **llama.cpp source:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\`
> **Build:** `E:\AI\LLM\llama.cpp\llama-server.exe`

## Token & Tokenizer

Model tidak baca huruf. Tokenizer mengubah teks jadi deretan angka (token) dan sebaliknya.

```
"Campurkan 2 telur"
  → tokenizer potong: ["Campur", "kan", " ", "2", " telur"]
  → jadi angka: [1243, 567, 1, 331, 8892]
```

## Forward Pass

Satu kali proses data dari input ke output melalui semua layer neural network.

```
Input → Layer1 → Layer2 → ... → LayerN → Output
```

Tidak bisa lompat layer. Setiap layer bergantung pada output layer sebelumnya.

## Hidden State

Representasi internal model di suatu layer. Berupa vektor float dengan dimensi `d_model` (misal 2048).

```
Layer N → [0.9, -0.8, 0.2, 0.7, -0.4, ...] ← hidden state
            ↑ 2048 angka float (dimensi, BUKAN jumlah token)
```

Hidden state adalah "pemikiran abstrak" model — **tidak berisi calon token**. Ia representasi terkompresi dari pemahaman model pada titik itu.

### Hidden state vs Output

| | Hidden state | Output |
|---|---|---|
| Bentuk | Vektor float dengan dimensi d_model (misal 2048) | Vektor probabilitas dengan panjang vocab (misal 248320) |
| Dimensi | d_model (model-dependent: 768, 2048, 4096, 5120) | vocab_size (jumlah token di vocabulary) |
| Isi | Pemikiran abstrak — TIDAK berisi token | Probabilitas setiap token di vocabulary |
| Dibaca manusia? | Tidak | Ya (interpretasi) |

## lm_head

**BUKAN otak model.** Hanya matriks proyeksi linear (1 lapisan) yang memetakan hidden state ke vocabulary.

```
hidden state (2048) → lm_head (2048 × 248320) → probabilitas (248320)
                       ↑ BUKAN otak, hanya mapping linear
```

### Cara kerja
Setiap token di vocabulary punya vektor bobot sendiri di matriks lm_head. Hasilnya adalah **dot product** antara hidden state dengan vektor setiap token:

```
probabilitas[token_i] = dot(hidden_state, lm_head[token_i])
```

Semakin cocok hidden state dengan vektor token_i → semakin tinggi probabilitasnya.

### Siapa yang menentukan bobot lm_head?
**Training.** Bobot lm_head adalah parameter yang di-**learn** lewat backpropagation, sama seperti parameter layer lainnya. Awalnya random, lalu dikoreksi sedikit demi sedikit setiap kali model salah tebak token. Tidak ada yang membuat manual.

### lm_head ≠ otak
- Transformer layers = **koki** (memasak, memproses, memahami konteks)
- lm_head = **piring** (hanya menyajikan hasil akhir, tidak memasak)

### lm_head ≠ Hidden State Dimension
Jangan bingung antara dimensi hidden state (d_model) dan vocab_size:
- d_model (2048) = dimensi vektor hidden state
- vocab_size (248320) = jumlah baris di matriks lm_head
- lm_head = matriks berukuran d_model × vocab_size

Setiap model punya satu lm_head. MTP punya lm_head terpisah untuk setiap prediksi tambahan.

## Autoregressive

Model generate token satu per satu, setiap langkah hanya melihat token sebelumnya.

```
Input: "Tulis"
  ↓ forward pass
Output probabilitas: "fungsi" 80%, "cerita" 10%, ...
  ↓ pilih tertinggi
Hasil: "Tulis fungsi"
  ↓ forward pass lagi
Output: "fibonacci" 75%, ...
```

Model tidak tahu berapa langkah lagi yang dibutuhkan — cuma nebak 1 token ke depan terus-menerus.

## Prefill vs Decode

### Prefill
Memproses prompt awal — semua token diproses paralel dalam 1 forward pass. Cepat.

### Decode
Generate token satu per satu. Lambat karena cuma 1 token per forward pass.

## Speculative Decoding

Teknik mempercepat decode dengan cara:
1. Model kecil (draft) generate beberapa token cepat
2. Model besar (target) verify token tersebut dalam 1 batch
3. Kalau cocok → dapat gratis, kalau tidak → buang dan lanjut

### Analogi
Tanpa speculative decoding = beli tiap langkah Rp10.000.
Speculative decoding = bayar Rp10.000 sekali, dapat draf 3 langkah murah, verify 3 token barengan.

### Jenis speculative decoding:
- **MTP** — head built-in di GGUF yang sama (Qwen3.6)
- **Draft model** — model terpisah (kecil) sebagai drafter
- **EAGLE** — drafter berdasarkan hidden state
- **N-gram** — berdasarkan pola berulang di prompt
- **Self-speculative** — berdasarkan history token
