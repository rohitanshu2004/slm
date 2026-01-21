import os
from pathlib import Path

class Settings:
    # App settings
    APP_NAME: str = "Financial RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # File settings
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = ['.pdf', '.xlsx', '.xls', '.csv']
    UPLOAD_DIR: str = "temp/uploads"
    PROCESSED_DIR: str = "temp/processed"

    # Model settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en"
    SLM_MODEL: str = "microsoft/phi-2"  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Vector store settings
    VECTOR_DB_PATH: str = "data/vector_db"
    COLLECTION_NAME: str = "financial_documents"

    # Chunking settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    def __init__(self):
        # Load from environment variables if available
        self.DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
        self.EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', self.EMBEDDING_MODEL)
        self.SLM_MODEL = os.getenv('SLM_MODEL', self.SLM_MODEL)
        self.VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', self.VECTOR_DB_PATH)
        self.COLLECTION_NAME = os.getenv('COLLECTION_NAME', self.COLLECTION_NAME)
        self.CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', self.CHUNK_OVERLAP))
        self.TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', self.TOP_K_RESULTS))
        self.SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', self.SIMILARITY_THRESHOLD))

settings = Settings()

# Create directories
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
