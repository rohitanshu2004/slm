# Financial Document Q&A - Utility Scripts

Command-line utilities for interacting with the Financial Document Q&A RAG system.

## Prerequisites

1. **Backend must be running**:
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

2. **Ollama must be running** with required models:
   ```bash
   ollama serve
   ollama pull nomic-embed-text
   ollama pull llama2:7b
   ```

3. **Python dependencies** (already in requirements.txt):
   - `requests`
   - `colorama`

## Scripts

### 1. Upload Documents (`upload_documents.py`)

Upload documents to the RAG system for processing and indexing.

#### Basic Usage

```bash
# Upload a single file
python scripts/upload_documents.py document.pdf

# Upload multiple files
python scripts/upload_documents.py file1.pdf file2.xlsx file3.csv

# Upload with glob pattern
python scripts/upload_documents.py documents/*.pdf

# Upload all PDFs in a directory
python scripts/upload_documents.py reports/*.pdf data/*.pdf
```

#### Advanced Options

```bash
# Clear existing documents before upload
python scripts/upload_documents.py --clear documents/*.pdf

# Use custom backend URL
python scripts/upload_documents.py --backend-url http://192.168.1.100:8001 document.pdf

# Show help
python scripts/upload_documents.py --help
```

#### Supported File Types

- **PDF** (`.pdf`) - Extracted using pdfplumber and PyPDF2
- **Excel** (`.xlsx`, `.xls`) - Processed with pandas and openpyxl
- **CSV** (`.csv`) - Processed with pandas

#### File Limits

- **Max file size**: 50 MB per file
- **Max total upload**: No limit on number of files (but consider processing time)

#### Example Output

```
============================================================
  Financial Document Upload Utility
============================================================

Checking backend at http://localhost:8001...
✓ Backend is online

Validating 3 file(s)...
  ✓ financial_report_2023.pdf
  ✓ budget_analysis.xlsx
  ✓ sales_data.csv

Preparing 3 file(s) for upload...
  ✓ financial_report_2023.pdf (2.45 MB)
  ✓ budget_analysis.xlsx (156.78 KB)
  ✓ sales_data.csv (45.23 KB)

Total size: 2.64 MB

Uploading to http://localhost:8001/upload...

============================================================
✓ Upload Successful!
============================================================

Files processed: 3
Chunks created: 87
Processing time: 12.34s

✓ All done!
```

---

### 2. Terminal Chat (`terminal_chat.py`)

Interactive terminal-based chat interface for querying documents.

#### Basic Usage

```bash
# Start interactive chat
python scripts/terminal_chat.py

# Use custom backend URL
python scripts/terminal_chat.py --backend-url http://192.168.1.100:8001
```

#### Available Commands

Once in the chat interface, you can use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/stats` | Display vector store statistics (document count, status) |
| `/clear` | Clear all documents from vector store |
| `/history` | Show conversation history for current session |
| `/exit` or `/quit` | Exit the chat interface |

#### Example Session

```
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       📚 Financial Document Q&A - Terminal Chat 💬           ║
    ║                                                               ║
    ║       Powered by Ollama + FAISS + LangChain                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝

Type your questions about uploaded documents.
Type /help for available commands or /exit to quit.

Connecting to backend at http://localhost:8001...
✓ Connected to backend

============================================================
  Vector Store Statistics
============================================================

Collection: financial_documents
Documents: 87
Status: active


You: What were the total revenues in Q4 2023?

Answer:
Based on the financial documents, the total revenues for Q4 2023 were 
$45.2 million, representing a 12% increase compared to Q4 2022.

Sources:

  Source 1 [Relevance: 92%]
    File: financial_report_2023.pdf (PDF)
    Page: 15
    Preview: "Q4 2023 Financial Results: Total revenues reached $45.2M..."

  Source 2 [Relevance: 87%]
    File: quarterly_summary.xlsx (XLSX)
    Rows: 45-50
    Preview: "Revenue breakdown by quarter showing Q4 at $45.2M..."

Processing time: 8.42s | Tokens: 234 | Confidence: 92%


You: /stats

============================================================
  Vector Store Statistics
============================================================

Collection: financial_documents
Documents: 87
Status: active


You: /exit

Goodbye! 👋
```

#### Query Tips

- **Be specific**: "What were the revenues in Q4 2023?" works better than "revenues"
- **Ask questions naturally**: The system understands natural language
- **Check sources**: Review the source documents and relevance scores
- **Use commands**: Type `/stats` to verify documents are loaded

---

## Troubleshooting

### Backend Not Running

**Error**: `✗ Cannot connect to backend at http://localhost:8001`

**Solution**: Start the backend server:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

### Ollama Not Available

**Error**: Backend responds with errors about Ollama models

**Solution**: 
1. Start Ollama: `ollama serve`
2. Pull required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama2:7b
   ```

### No Documents Found

**Error**: `✗ No documents found in vector store`

**Solution**: Upload documents first:
```bash
python scripts/upload_documents.py documents/*.pdf
```

### File Validation Errors

**Error**: `File too large` or `Unsupported file type`

**Solution**: 
- Ensure files are under 50 MB
- Only upload `.pdf`, `.xlsx`, `.xls`, or `.csv` files
- Check file is not corrupted

### Upload Timeout

**Error**: `✗ Upload timed out`

**Solution**: 
- Upload fewer files at once
- Reduce file sizes
- Check backend logs for processing errors

### Missing Dependencies

**Error**: `ModuleNotFoundError: No module named 'requests'`

**Solution**: Install dependencies:
```bash
pip install requests colorama
```

Or install all backend requirements:
```bash
cd backend
pip install -r requirements.txt
```

---

## Workflow Examples

### Initial Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Start backend (in separate terminal)
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001

# 3. Upload documents
python scripts/upload_documents.py --clear documents/*.pdf

# 4. Start chatting
python scripts/terminal_chat.py
```

### Daily Usage

```bash
# Add new documents (keeping existing ones)
python scripts/upload_documents.py new_report.pdf

# Query documents
python scripts/terminal_chat.py
```

### Reset and Start Fresh

```bash
# Clear all documents and upload new ones
python scripts/upload_documents.py --clear documents/*.pdf
```

---

## Technical Details

### Upload Process

1. Files are validated for extension and size
2. Files are read and sent as multipart/form-data
3. Backend processes each file:
   - Extracts text/tables
   - Creates semantic chunks
   - Generates embeddings using Ollama
   - Stores in FAISS vector database
4. Returns processing statistics

### Query Process

1. User query is sent to backend
2. Backend generates query embedding using Ollama
3. FAISS performs similarity search
4. Top-K most relevant chunks are retrieved
5. Ollama generates answer using context
6. Answer and sources are returned to terminal

### Configuration

Default settings (can be modified in script files):
- **Backend URL**: `http://localhost:8001`
- **Upload timeout**: 300 seconds (5 minutes)
- **Query timeout**: 60 seconds
- **Top-K results**: 5
- **Similarity threshold**: 0.7 (70%)
- **Max file size**: 50 MB

---

## Advanced Usage

### Custom Backend URL

If your backend is on a different machine or port:

```bash
python scripts/upload_documents.py --backend-url http://192.168.1.100:8001 docs/*.pdf
python scripts/terminal_chat.py --backend-url http://192.168.1.100:8001
```

### Batch Processing

Process multiple directories:

```bash
python scripts/upload_documents.py \
    reports/2023/*.pdf \
    reports/2024/*.pdf \
    budgets/*.xlsx \
    data/*.csv
```

### Automation

Upload all documents in a directory automatically:

```bash
# Linux/Mac
find documents/ -type f \( -name "*.pdf" -o -name "*.xlsx" -o -name "*.csv" \) \
    -exec python scripts/upload_documents.py {} +

# Windows PowerShell
Get-ChildItem -Path documents -Include *.pdf,*.xlsx,*.csv -Recurse | 
    ForEach-Object { python scripts/upload_documents.py $_.FullName }
```

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend logs: Check the terminal running uvicorn
3. Check Ollama logs: `ollama logs`
4. Verify all services are running: backend, Ollama

---

## License

Part of the Financial Document Q&A system.
