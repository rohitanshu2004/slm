# Step-by-Step Guide to Start the RAG Application with Ollama

## Prerequisites
- Python 3.8+
- Ollama installed and running

## Step 1: Install Ollama
Download and install Ollama from: https://ollama.ai/download

### For Windows:
1. Download the installer from the Ollama website
2. Run the installer
3. Ollama will be installed and the service will start automatically

### For other platforms:
Follow the installation instructions on the Ollama website for your operating system.

## Step 2: Pull Required Models
Open a terminal/command prompt and run:

```bash
# Pull the embedding model (optimized for semantic search)
ollama pull nomic-embed-text

# Pull the text generation model
ollama pull llama2:7b

# Optional: Pull other models if you want to use different ones
ollama pull llama2:13b  # Larger model for better quality
ollama pull mistral     # Alternative model
```

## Step 3: Verify Ollama Installation
```bash
# Check if Ollama is running and models are available
ollama list

# Test the embedding model
ollama run nomic-embed-text "Generate embedding for this text"

# Test the generation model
ollama run llama2:7b "Hello, how are you?"
```

## Step 4: Install Python Dependencies
After setting up Ollama and pulling the required models, install the Python dependencies:

```bash
pip install -r backend/requirements.txt
```

## Step 5: Configure Environment (Optional)
Create a `.env` file in the backend directory if you need to customize settings:

```env
# Ollama API settings
OLLAMA_API_URL=http://localhost:11434
OLLAMA_API_KEY=  # Leave empty if not required

# Model settings
EMBEDDING_MODEL=nomic-embed-text  # Optimized for embeddings
SLM_MODEL=llama2:7b              # For text generation

# Other settings
DEBUG=true
VECTOR_DB_PATH=data/vector_db
```

## Step 6: Start the Backend Server
```bash
cd backend
python main.py
```

The server should start on `http://localhost:8000`

## Step 7: Start the Frontend (Optional)
If you have a frontend component:

```bash
cd frontend
python app.py
```

## Step 8: Test the Application
You can test the API endpoints using curl or a tool like Postman:

### Upload a document:
```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

### Query the RAG system:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the main topic of the document?"}'
```

## Troubleshooting

### Ollama Connection Issues:
1. Make sure Ollama is running: `ollama serve`
2. Check if the API URL in config.py matches your Ollama setup
3. Verify the model is pulled: `ollama list`

### Python Dependencies:
1. Make sure you're in the correct virtual environment
2. Try reinstalling: `pip install -r requirements.txt --force-reinstall`

### Port Issues:
- If port 8000 is busy, modify the port in `main.py`
- If port 11434 (Ollama default) is busy, change `OLLAMA_API_URL` in config

## Available Models

### Embedding Models (for semantic search):
- `nomic-embed-text` - Recommended, optimized for embeddings
- `mxbai-embed-large` - Alternative embedding model
- `all-minilm` - Lightweight embedding model

### Generation Models (for text responses):
- `llama2:7b` - Fast, good quality (recommended)
- `llama2:13b` - Better quality, slower
- `mistral` - Alternative model
- `codellama` - Code-focused model

## Performance Tips
1. Use smaller models for faster inference
2. Keep Ollama running in the background
3. Monitor system resources when processing large documents
