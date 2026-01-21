import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import logging

from ..config import settings

logger = logging.getLogger(__name__)

class CSVProcessor:
    def __init__(self, max_rows_per_chunk: int = 100):
        self.max_rows_per_chunk = max_rows_per_chunk
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file"""
        chunks = []
        file_name = Path(file_path).name
        
        try:
            # Read CSV with flexible options
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                except:
                    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            
            # Clean dataframe
            df = self._clean_dataframe(df)
            
            # Process dataframe into chunks
            chunks = self._process_dataframe(df, file_name)
            
        except Exception as e:
            logger.error(f"Error processing CSV file {file_name}: {e}")
            raise Exception(f"Failed to process CSV file: {e}")
        
        return chunks
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe"""
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Fill NaN values
        df = df.fillna('')
        
        # Convert all to string for consistency
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        return df
    
    def _process_dataframe(self, df: pd.DataFrame, file_name: str) -> List[Dict]:
        """Process dataframe into chunks"""
        chunks = []
        
        # Create header info
        header_info = f"CSV File: {file_name}\n"
        header_info += f"Dimensions: {len(df)} rows × {len(df.columns)} columns\n"
        
        if len(df.columns) <= 15:
            header_info += f"Columns: {', '.join(df.columns)}\n"
        
        # Handle different sizes
        if len(df) <= self.max_rows_per_chunk:
            # Small file - single chunk
            markdown_table = df.to_markdown(index=False)
            
            chunks.append({
                'content': header_info + "\n" + markdown_table,
                'metadata': {
                    'file_type': 'csv',
                    'rows': len(df),
                    'columns': list(df.columns),
                    'source': file_name
                }
            })
        else:
            # Large file - split into chunks
            num_chunks = (len(df) + self.max_rows_per_chunk - 1) // self.max_rows_per_chunk
            
            for i in range(num_chunks):
                start_idx = i * self.max_rows_per_chunk
                end_idx = min((i + 1) * self.max_rows_per_chunk, len(df))
                
                chunk_df = df.iloc[start_idx:end_idx]
                markdown_table = chunk_df.to_markdown(index=False)
                
                chunk_header = f"{header_info}Rows {start_idx + 1} to {end_idx} of {len(df)}:\n"
                
                chunks.append({
                    'content': chunk_header + markdown_table,
                    'metadata': {
                        'file_type': 'csv',
                        'rows': f"{start_idx + 1}-{end_idx}",
                        'total_rows': len(df),
                        'columns': list(df.columns),
                        'source': file_name,
                        'chunk_number': i + 1,
                        'total_chunks': num_chunks
                    }
                })
        
        return chunks