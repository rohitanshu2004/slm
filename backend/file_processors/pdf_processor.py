import pdfplumber
import PyPDF2
from typing import List, Dict, Any
import re
from pathlib import Path
import logging
from config import settings
from exceptions import FileProcessingException

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PDF file and extract text with metadata"""
        chunks = []
        file_name = Path(file_path).name
        # Enhanced validation: file exists, not empty, and valid PDF
        p = Path(file_path)
        if not p.exists() or p.stat().st_size == 0:
            logger.error(f"PDF file missing or empty: {file_path}")
            raise FileProcessingException(
                message="PDF file is missing or empty",
                filename=file_name,
                file_type='pdf'
            )

        # Validate PDF structure before processing
        if not self._validate_pdf_structure(file_path, file_name):
            raise FileProcessingException(
                message="PDF file is corrupted or has invalid structure",
                filename=file_name,
                file_type='pdf'
            )

        try:
            # Try using pdfplumber first (better for text extraction)
            chunks.extend(self._process_with_pdfplumber(file_path, file_name))
        except Exception as e1:
            logger.warning(f"pdfplumber failed: {e1}. Trying PyPDF2...")
            try:
                chunks.extend(self._process_with_pypdf2(file_path, file_name))
            except Exception as e2:
                logger.error(f"Both PDF processors failed: {e2}")
                raise FileProcessingException(
                    message="Failed to process PDF with available parsers",
                    filename=file_name,
                    file_type='pdf',
                    details={"error": str(e2)}
                )

        if not chunks:
            logger.error(f"No text could be extracted from PDF: {file_name}")
            raise FileProcessingException(
                message="No extractable text found in PDF file",
                filename=file_name,
                file_type='pdf'
            )

        return chunks
    
    def _process_with_pdfplumber(self, file_path: str, file_name: str) -> List[Dict]:
        """Process PDF using pdfplumber"""
        chunks = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text()
                if text:
                    # Clean text
                    text = self._clean_text(text)
                    
                    # Create chunks from page text
                    page_chunks = self._chunk_text(text, page_num, file_name, 'text')
                    chunks.extend(page_chunks)
                
                # Extract tables
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, 1):
                    if table:
                        table_text = self._table_to_markdown(table)
                        if table_text:
                            chunks.append({
                                'content': table_text,
                                'metadata': {
                                    'file_type': 'pdf',
                                    'page_number': page_num,
                                    'table_number': table_num,
                                    'content_type': 'table',
                                    'source': file_name
                                }
                            })
        
        return chunks
    
    def _process_with_pypdf2(self, file_path: str, file_name: str) -> List[Dict]:
        """Fallback to PyPDF2 if pdfplumber fails"""
        chunks = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text:
                    text = self._clean_text(text)
                    page_chunks = self._chunk_text(text, page_num + 1, file_name, 'text')
                    chunks.extend(page_chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR/text extraction issues
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix spaced numbers
        text = re.sub(r'([a-zA-Z])-\s+([a-zA-Z])', r'\1\2', text)  # Fix hyphenated words
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def _chunk_text(self, text: str, page_num: int, file_name: str, content_type: str) -> List[Dict]:
        """Split text into chunks"""
        chunks = []
        
        # Split by sentences for better semantic boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'file_type': 'pdf',
                            'page_number': page_num,
                            'content_type': content_type,
                            'source': file_name
                        }
                    })
                current_chunk = sentence + " "
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'file_type': 'pdf',
                    'page_number': page_num,
                    'content_type': content_type,
                    'source': file_name
                }
            })
        
        return chunks
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """Convert table to markdown format"""
        if not table or not any(table):
            return ""
        
        # Clean table data
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Convert to string and clean
                    cell_str = str(cell).strip()
                    # Escape pipe characters
                    cell_str = cell_str.replace('|', '\\|')
                    cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        # Create markdown table
        md_lines = []
        
        # Header row
        if cleaned_table:
            md_lines.append("| " + " | ".join(cleaned_table[0]) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(cleaned_table[0])) + " |")
            
            # Data rows
            for row in cleaned_table[1:]:
                if row:  # Skip empty rows
                    md_lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(md_lines)