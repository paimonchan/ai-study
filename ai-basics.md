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

Representasi internal model di suatu layer. Berupa vektor float (misal 2048 angka).

```
Layer N → [0.9, -0.8, 0.2, 0.7, -0.4, ...] ← hidden state
```

Hidden state adalah "pemikiran abstrak" model — belum diterjemahkan ke token.

### Hidden state vs Output

| | Hidden state | Output |
|---|---|---|
| Bentuk | Vektor float (misal 2048) | Vektor probabilitas (misal 248320) |
| Isi | Pemikiran abstrak | Prediksi token |
| Dibaca manusia? | Tidak | Ya (interpretasi) |

## lm_head

Fungsi/matriks yang mengubah hidden state jadi probabilitas untuk setiap token di vocabulary.

```
hidden state (2048) → lm_head (2048 × 248320) → probabilitas (248320)
```

Setiap model punya lm_head sendiri. MTP punya lm_head terpisah.

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

### Jenis speculative decoding:
- **MTP** — head built-in di GGUF yang sama (Qwen3.6)
- **Draft model** — model terpisah (kecil) sebagai drafter
- **EAGLE** — drafter berdasarkan hidden state
- **N-gram** — berdasarkan pola berulang di prompt
- **Self-speculative** — berdasarkan history token
