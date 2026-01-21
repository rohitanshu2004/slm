import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging

from ..config import settings

logger = logging.getLogger(__name__)

class ExcelProcessor:
    def __init__(self, max_rows_per_chunk: int = 50):
        self.max_rows_per_chunk = max_rows_per_chunk
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Excel file with multiple sheets"""
        chunks = []
        file_name = Path(file_path).name
        
        try:
            # Try reading as Excel
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            
            for sheet_name in sheet_names:
                try:
                    # Read sheet
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    # Process the sheet
                    sheet_chunks = self._process_sheet(df, sheet_name, file_name)
                    chunks.extend(sheet_chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing sheet {sheet_name}: {e}")
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
        
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise Exception(f"Failed to process Excel file: {e}")
        
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