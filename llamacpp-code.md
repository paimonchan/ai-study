# llama.cpp Code Structure

> **Source code lokal:** `E:\AI\LLM\_work\llama.cpp-upstream-src-tmp\`
> (clone dari `github.com/ggml-org/llama.cpp`, depth 1, bisa dibaca langsung)
>
> **Build directory:** `E:\AI\LLM\_work\llama.cpp-upstream-build-cuda\`
>
> **Binary:** `E:\AI\LLM\llama.cpp\llama-server.exe`

## Server Layer (tools/server/)

| File | Baris | Fungsi |
|---|---|---|
| `server.cpp` | 313 | Entry point, routing, main() |
| `server-http.cpp` | 27k | HTTP server, request → response |
| `server-context.cpp` | 193k | **Inti** — inference loop, slot management |
| `server-chat.cpp` | 27k | Chat template formatting |
| `server-task.cpp` | 85k | Task queue processing |
| `server-models.cpp` | 61k | Model loading, router mode |
| `server-common.cpp` | 60k | Helpers, utils |
| `server-tools.cpp` | 30k | Built-in tools |

### Flow server.cpp main()
```
1. common_params_parse()     ← parsing CLI args
2. ctx_http.init(params)     ← start HTTP server
3. ctx_server.load_model()   ← load model + MTP
4. routes.post_*             ← register endpoint handler
5. ctx_server.start_loop()   ← main inference loop (blocking)
```

## MTP Layer (common/)

| File | Baris | Fungsi |
|---|---|---|
| `speculative.h` | ~3k | API declarations |
| `speculative.cpp` | 55k | All speculative methods (MTP, draft, n-gram, etc) |

### MTP state machine (speculative.cpp:380)
```
struct common_speculative_state_draft_mtp :
    public common_speculative_impl {

    process()  → feed hidden state target ke MTP
    draft()    → generate draft tokens
    accept()   → verify & rollback
}
```

### API functions
```
common_speculative_init()     — create speculative state
common_speculative_process()  — feed hidden states (after target decode)
common_speculative_draft()    — generate draft tokens
common_speculative_accept()   — accept/reject verified tokens
common_speculative_print_stats() — print statistics
```

## Build Setup

### Prerequisites
| Tool | Path |
|---|---|
| CUDA Toolkit DLLs | `E:\AI\LLM\_work\cuda-toolchain\Library\bin` |
| nvcc | `E:\AI\LLM\_work\cuda-toolchain\Library\bin\nvcc.exe` |
| CMake | `E:\AI\LLM\cmake-portable\cmake-3.31.6-windows-x86_64\bin\cmake.exe` |
| Ninja | `D:\DataDev\anaconda3\Library\bin\ninja.exe` |
| MSVC | `D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64\cl.exe` |
| Windows SDK | `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\` |

### Build step
```powershell
# Set environment (wajib!)
$env:PATH = "E:\AI\LLM\_work\cuda-toolchain\Library\bin;D:\DataDev\anaconda3\Library\bin;D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64;C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64;" + $env:PATH
$env:INCLUDE = "D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\shared;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\um;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\winrt"
$env:LIB = "D:\Data NIP\Dev\Microsoft Visual Studio\VC\Tools\MSVC\14.39.33519\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

# Clone
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git E:\AI\LLM\llama.cpp-src

# Configure (pakai Ninja, bukan VS generator!)
& "E:\AI\LLM\cmake-portable\cmake-3.31.6-windows-x86_64\bin\cmake.exe" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 -S "E:\AI\LLM\llama.cpp-src" -B "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda"

# Build (hanya llama-server, ~10-15 menit)
& "D:\DataDev\anaconda3\Library\bin\ninja.exe" -C "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda" llama-server

# Copy ke stable
Copy-Item "E:\AI\LLM\_work\llama.cpp-upstream-build-cuda\bin\*" "E:\AI\LLM\llama.cpp\" -Force
```

### Catatan penting
- **Pakai Ninja**, jangan Visual Studio generator — CUDA toolchain portable tidak punya VS integration files
- **Set PATH, INCLUDE, LIB** — tanpa ini Ninja + MSVC gagal link
- **Build hanya `llama-server`** — `ninja all` butuh ~1 jam
- **CUDA DLLs di PATH** — `llama-server.exe` butuh `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`

### Perbedaan fork vs upstream
| Aspek | Fork (am17an/mtp-clean) | Upstream master |
|---|---|---|
| Arg MTP | `--spec-type mtp` | `--spec-type draft-mtp` |
| Binary size | 8.7 MB | 7.5 MB |
| Speed (35B IQ2_M) | 58.1 t/s | 83.0 t/s |
