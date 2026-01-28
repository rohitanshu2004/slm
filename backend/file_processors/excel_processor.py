import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging

from config import settings

logger = logging.getLogger(__name__)
from exceptions import FileProcessingException

class ExcelProcessor:
    def __init__(self, max_rows_per_chunk: int = 50):
        self.max_rows_per_chunk = max_rows_per_chunk
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Excel file with multiple sheets"""
        chunks = []
        file_name = Path(file_path).name
        # Validate file
        p = Path(file_path)
        if not p.exists() or p.stat().st_size == 0:
            logger.error(f"Excel file missing or empty: {file_path}")
            raise FileProcessingException(
                message="Excel file is missing or empty",
                filename=file_name,
                file_type='excel'
            )

        try:
            # Try reading as Excel
            try:
                xls = pd.ExcelFile(file_path)
                sheet_names = xls.sheet_names
            except Exception as e:
                logger.exception("Failed to open Excel file")
                raise FileProcessingException(
                    message="Failed to read Excel file",
                    filename=file_name,
                    file_type='excel',
                    details={"error": str(e)}
                )

            for sheet_name in sheet_names:
                try:
                    # Read sheet
                    df = pd.read_excel(xls, sheet_name=sheet_name)

                    # Process the sheet
                    sheet_chunks = self._process_sheet(df, sheet_name, file_name)
                    chunks.extend(sheet_chunks)

                except FileProcessingException:
                    # Sheet-level processing raised a structured error; re-raise
                    raise
                except Exception as e:
                    logger.exception(f"Error processing sheet {sheet_name}")
                    # Create a simple chunk with error info
                    chunks.append({
                        'content': f"Sheet: {sheet_name}\nError: Could not process this sheet.",
                        'metadata': {
                            'file_type': 'excel',
                            'sheet_name': sheet_name,
                            'rows': 0,
                            'columns': [],
                            'source': file_name,
                            'error': str(e)
                        }
                    })

        except FileProcessingException:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error processing Excel file {file_name}")
            raise FileProcessingException(
                message="Failed to process Excel file",
                filename=file_name,
                file_type='excel',
                details={"error": str(e)}
            )

        if not chunks:
            logger.error(f"No usable data extracted from Excel file: {file_name}")
            raise FileProcessingException(
                message="Excel file contains no usable data",
                filename=file_name,
                file_type='excel'
            )

        return chunks
    
    def _process_sheet(self, df: pd.DataFrame, sheet_name: str, file_name: str) -> List[Dict]:
        """Process a single sheet into chunks"""
        chunks = []
        
        # Clean dataframe
        df = self._clean_dataframe(df)
        
        # Create descriptive header
        header_info = self._create_header_info(df, sheet_name)
        
        # Handle empty or very small dataframes
        if len(df) == 0:
            chunks.append({
                'content': header_info + "\nThis sheet is empty.",
                'metadata': {
                    'file_type': 'excel',
                    'sheet_name': sheet_name,
                    'rows': 0,
                    'columns': list(df.columns),
                    'source': file_name
                }
            })
            return chunks
        
        # Split large dataframes into chunks
        if len(df) <= self.max_rows_per_chunk:
            # Convert entire dataframe to markdown
            markdown_table = df.to_markdown(index=False)
            
            chunks.append({
                'content': header_info + "\n" + markdown_table,
                'metadata': {
                    'file_type': 'excel',
                    'sheet_name': sheet_name,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'source': file_name
                }
            })
        else:
            # Split dataframe into chunks
            num_chunks = (len(df) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk
            
            for i in range(num_chunks):
                start_idx = i * self.max_rows_per_chunk
                end_idx = min((i + 1) * self.max_rows_per_chunk, len(df))
                
                chunk_df = df.iloc[start_idx:end_idx]
                markdown_table = chunk_df.to_markdown(index=False)
                
                chunk_header = f"{header_info}\nRows {start_idx + 1} to {end_idx} of {len(df)}:\n"
                
                chunks.append({
                    'content': chunk_header + markdown_table,
                    'metadata': {
                        'file_type': 'excel',
                        'sheet_name': sheet_name,
                        'rows': f"{start_idx + 1}-{end_idx}",
                        'total_rows': len(df),
                        'columns': list(df.columns),
                        'source': file_name,
                        'chunk_number': i + 1,
                        'total_chunks': num_chunks
                    }
                })
        
        return chunks
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataframe for better processing"""
        # Make a copy
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Fill NaN with empty string for better markdown conversion
        df = df.fillna('')
        
        # Convert all columns to string for consistency
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        return df
    
    def _create_header_info(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Create descriptive header for the sheet"""
        header = f"Excel Sheet: {sheet_name}\n"
        header += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n"

        if len(df.columns) <= 10:  # Only show columns if not too many
            header += f"Columns: {', '.join(df.columns)}\n"
        else:
            header += f"Columns: {len(df.columns)} columns (too many to list)\n"

        # Add sample data if available
        if len(df) > 0:
            header += f"First few rows show: {df.iloc[0, 0][:50]}...\n"

        return header

    def _validate_excel_structure(self, file_path: str, file_name: str) -> bool:
        """Validate Excel file structure before processing"""
        try:
            # Check file header to ensure it's an Excel file
            with open(file_path, 'rb') as f:
                header = f.read(8)

            # Excel files start with specific signatures
            excel_signatures = [
                b'\x50\x4B\x03\x04',  # ZIP/XLSX signature
                b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1',  # XLS signature (OLE2)
            ]

            if not any(header.startswith(sig) for sig in excel_signatures):
                logger.error(f"File {file_name} does not have valid Excel signature")
                return False

            # Try to open with pandas to validate structure
            try:
                xls = pd.ExcelFile(file_path)
                if not xls.sheet_names:
                    logger.error(f"Excel file {file_name} has no sheets")
                    return False
            except Exception as e:
                logger.error(f"Failed to validate Excel structure for {file_name}: {e}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating Excel file {file_name}: {e}")
            return False
