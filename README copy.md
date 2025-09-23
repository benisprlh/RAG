# RAG Combined App

Aplikasi ini adalah contoh pipeline Retrieval Augmented Generation (RAG) modern yang menggabungkan berbagai teknologi.

## Ringkasan Fitur
### Sudah Diimplementasi
- [x] **/ask**: Tanya jawab dokumen berbasis RAG (vector search + LLM + semantic cache)
- [x] **/chat**: Multi-turn chat dengan memory per user (riwayat chat, context-aware)
- [x] **/upload**: Upload dokumen PDF baru, otomatis diindex ke Redis vectorstore
- [x] **Semantic cache**: Jawaban LLM untuk pertanyaan mirip di-cache otomatis
- [x] **Chat memory**: Riwayat chat per user disimpan dan digunakan sebagai context
- [x] **Advanced RedisVL**: Hybrid search, metadata/tag filter, range query, role-based retrieval

### Belum Diimplementasi (Coming Soon)
- [ ] **/evaluate**: Evaluasi pipeline dengan ragas (faithfulness, relevancy, recall, precision)
- [ ] **Query rewriting**: Penulisan ulang pertanyaan dengan LLM sebelum retrieval
- [ ] **Optimasi indexing incremental** (saat upload dokumen baru)

## Teknologi yang Digunakan
- **Redis**: Vector database utama
- **LangChain**: Orkestrasi pipeline modular (retriever, prompt, LLM, dsb)
- **redisvl**: Fitur Redis lanjutan (semantic cache, chat memory, dsb)
- **OpenAI & NVIDIA**: LLM dan embedding (GPT, Mixtral, dsb)

## Cara Set Environment Variable
Aplikasi ini membutuhkan dua environment variable utama:
- `REDIS_URL` : URL koneksi ke Redis Stack (default: `redis://localhost:6379`)
- `OPENAI_API_KEY` : API key OpenAI (wajib jika ingin pakai LLM OpenAI)

### **Linux/MacOS**
```bash
export REDIS_URL="redis://localhost:6379"
export OPENAI_API_KEY="sk-..."
```

### **Windows (Command Prompt)**
```cmd
set REDIS_URL=redis://localhost:6379
set OPENAI_API_KEY=sk-...
```

### **Windows (PowerShell)**
```powershell
$env:REDIS_URL = "redis://localhost:6379"
$env:OPENAI_API_KEY = "sk-..."
```

> **Tips:**
> - Pastikan Redis Stack sudah berjalan di alamat yang sesuai dengan `REDIS_URL`.
> - Anda bisa mendapatkan API key OpenAI di https://platform.openai.com/account/api-keys
> - Anda bisa menambahkan perintah export/set di file `.env` atau script startup Anda.

## Struktur Folder
- `app/` : Kode utama aplikasi (pipeline, API, dsb)
- `notebooks/` : Notebook demonstrasi dan eksperimen
- `tests/` : Unit test dan evaluasi
- `requirements.txt` : Daftar dependensi

## Cara Pakai
1. Install dependensi: `pip install -r requirements.txt`
2. Set environment variable seperti di atas
3. Jalankan aplikasi API: `uvicorn app.api:app --reload`
4. Buka dokumentasi otomatis di: [http://localhost:8000/docs](http://localhost:8000/docs)

## Endpoint API

### 1. Tanya Jawab Dokumen (RAG)
- **POST /ask**
- Request body:
```json
{
  "question": "What is the trend in the company's revenue and profit over the past few years?",
  "k": 3,
  "filters": {"role": "finance"},
  "distance_threshold": 0.3,
  "text_filter": "profit"
}
```
- Response:
```json
{
  "answer": "...jawaban dari LLM...",
  "context": ["...potongan dokumen...", "..."],
  "cached": false
}
```

### 2. Chat Multi-turn dengan Memory
- **POST /chat**
- Request body:
```json
{
  "user_id": "user123",
  "message": "Bagaimana tren revenue Nike?",
  "k": 3,
  "filters": {"role": "finance"},
  "distance_threshold": 0.3,
  "text_filter": "profit"
}
```
- Response:
```json
{
  "answer": "...jawaban dari LLM...",
  "chat_history": [
    {"role": "user", "content": "Bagaimana tren revenue Nike?"},
    {"role": "assistant", "content": "...jawaban..."}
  ],
  "cached": false
}
```

### 3. Upload Dokumen PDF Baru
- **POST /upload**
- Form-data: `file` (PDF)
- Response:
```json
{
  "msg": "File <nama_file.pdf> berhasil diupload dan diindex."
}
```

### 4. Root
- **GET /**
- Response: `{ "msg": "RAG Combined API. Lihat /docs untuk dokumentasi." }`

---

Aplikasi ini dapat dikembangkan lebih lanjut sesuai kebutuhan use-case Anda. 