#!/usr/bin/env python3
"""
Upload Documents to RAG Backend

Command-line utility to upload documents to the Financial Document Q&A system.
Supports PDF, Excel (xlsx, xls), and CSV files with batch processing capabilities.

Usage:
    python upload_documents.py document.pdf
    python upload_documents.py file1.pdf file2.xlsx --clear
    python upload_documents.py documents/*.pdf
"""

import argparse
import sys
import os
from pathlib import Path
from glob import glob
from typing import List, Tuple
import requests
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configuration
DEFAULT_BACKEND_URL = "http://localhost:8001"
UPLOAD_TIMEOUT = 300  # 5 minutes
HEALTH_TIMEOUT = 5
SUPPORTED_EXTENSIONS = ['.pdf', '.xlsx', '.xls', '.csv']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes


def print_colored(message: str, color: str = Fore.WHITE, style: str = Style.NORMAL):
    """Print colored message to console"""
    print(f"{style}{color}{message}{Style.RESET_ALL}")


def format_file_size(bytes_size: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def check_backend_health(backend_url: str) -> bool:
    """Check if backend is running and accessible"""
    try:
        response = requests.get(f"{backend_url}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            return True
        else:
            print_colored(f"Backend returned status {response.status_code}", Fore.YELLOW)
            return False
    except requests.exceptions.ConnectionError:
        print_colored(f"✗ Cannot connect to backend at {backend_url}", Fore.RED)
        print_colored("  Make sure the backend is running:", Fore.YELLOW)
        print_colored("  cd backend && uvicorn main:app --host 0.0.0.0 --port 8001", Fore.CYAN)
        return False
    except requests.exceptions.Timeout:
        print_colored("✗ Backend health check timed out", Fore.RED)
        return False
    except Exception as e:
        print_colored(f"✗ Error checking backend health: {e}", Fore.RED)
        return False


def validate_file(file_path: Path) -> Tuple[bool, str]:
    """
    Validate file extension and size
    Returns: (is_valid, error_message)
    """
    # Check if file exists
    if not file_path.exists():
        return False, f"File not found: {file_path}"
    
    if not file_path.is_file():
        return False, f"Not a file: {file_path}"
    
    # Check extension
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: {file_path.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    # Check file size
    file_size = file_path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large: {format_file_size(file_size)} (max: {format_file_size(MAX_FILE_SIZE)})"
    
    if file_size == 0:
        return False, f"File is empty: {file_path}"
    
    return True, ""


def get_mime_type(file_path: Path) -> str:
    """Get MIME type for file based on extension"""
    mime_types = {
        '.pdf': 'application/pdf',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.csv': 'text/csv'
    }
    return mime_types.get(file_path.suffix.lower(), 'application/octet-stream')


def expand_file_patterns(patterns: List[str]) -> List[Path]:
    """Expand glob patterns and return list of file paths"""
    all_files = []
    for pattern in patterns:
        # Check if it's a glob pattern
        if '*' in pattern or '?' in pattern:
            matched_files = glob(pattern, recursive=True)
            all_files.extend([Path(f) for f in matched_files if Path(f).is_file()])
        else:
            # Direct file path
            path = Path(pattern)
            if path.exists() and path.is_file():
                all_files.append(path)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def upload_files(file_paths: List[Path], backend_url: str, clear_before: bool) -> bool:
    """
    Upload files to backend
    Returns: True if successful, False otherwise
    """
    if not file_paths:
        print_colored("✗ No files to upload", Fore.RED)
        return False
    
    # Prepare files for upload
    files_data = []
    total_size = 0
    
    print_colored(f"\nPreparing {len(file_paths)} file(s) for upload...", Fore.CYAN)
    
    for file_path in file_paths:
        try:
            file_size = file_path.stat().st_size
            total_size += file_size
            
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
                mime_type = get_mime_type(file_path)
                files_data.append(("files", (file_path.name, file_bytes, mime_type)))
            
            print_colored(f"  ✓ {file_path.name} ({format_file_size(file_size)})", Fore.GREEN)
        except Exception as e:
            print_colored(f"  ✗ Failed to read {file_path.name}: {e}", Fore.RED)
            return False
    
    print_colored(f"\nTotal size: {format_file_size(total_size)}", Fore.CYAN)
    
    # Upload to backend
    try:
        url = f"{backend_url}/upload"
        params = {"clear_before_upload": str(clear_before).lower()}
        
        if clear_before:
            print_colored("\n⚠ Clearing existing documents before upload...", Fore.YELLOW)
        
        print_colored(f"\nUploading to {url}...", Fore.CYAN)
        
        response = requests.post(
            url,
            files=files_data,
            params=params,
            timeout=UPLOAD_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print_colored("\n" + "="*60, Fore.GREEN)
            print_colored("✓ Upload Successful!", Fore.GREEN, Style.BRIGHT)
            print_colored("="*60, Fore.GREEN)
            
            print_colored(f"\nFiles processed: {result.get('files_processed', len(file_paths))}", Fore.CYAN)
            print_colored(f"Chunks created: {result.get('chunks_created', 'N/A')}", Fore.CYAN)
            print_colored(f"Processing time: {result.get('processing_time', 'N/A'):.2f}s", Fore.CYAN)
            
            if 'message' in result:
                print_colored(f"\n{result['message']}", Fore.GREEN)
            
            return True
            
        elif response.status_code == 409:
            print_colored("\n✗ Upload already in progress", Fore.RED)
            print_colored("  Please wait for the current upload to complete", Fore.YELLOW)
            return False
            
        elif response.status_code in [400, 422]:
            print_colored(f"\n✗ Validation Error ({response.status_code})", Fore.RED)
            try:
                error_data = response.json()
                if 'detail' in error_data:
                    if isinstance(error_data['detail'], list):
                        for error in error_data['detail']:
                            print_colored(f"  - {error.get('msg', error)}", Fore.YELLOW)
                    else:
                        print_colored(f"  {error_data['detail']}", Fore.YELLOW)
            except:
                print_colored(f"  {response.text}", Fore.YELLOW)
            return False
            
        elif response.status_code == 500:
            print_colored(f"\n✗ Server Error ({response.status_code})", Fore.RED)
            try:
                error_data = response.json()
                print_colored(f"  {error_data.get('message', 'Internal server error')}", Fore.YELLOW)
                if 'details' in error_data:
                    print_colored(f"  Details: {error_data['details']}", Fore.YELLOW)
            except:
                print_colored(f"  {response.text[:200]}", Fore.YELLOW)
            return False
            
        else:
            print_colored(f"\n✗ Unexpected error: HTTP {response.status_code}", Fore.RED)
            print_colored(f"  {response.text[:200]}", Fore.YELLOW)
            return False
            
    except requests.exceptions.Timeout:
        print_colored("\n✗ Upload timed out", Fore.RED)
        print_colored("  Try uploading fewer files or smaller files", Fore.YELLOW)
        return False
        
    except requests.exceptions.ConnectionError:
        print_colored("\n✗ Connection error", Fore.RED)
        print_colored("  Backend may have crashed or is not responding", Fore.YELLOW)
        return False
        
    except Exception as e:
        print_colored(f"\n✗ Upload failed: {e}", Fore.RED)
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Upload documents to the Financial Document Q&A system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s file1.pdf file2.xlsx file3.csv
  %(prog)s documents/*.pdf
  %(prog)s --clear documents/*.pdf
  %(prog)s --backend-url http://192.168.1.100:8001 document.pdf
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Files to upload (supports glob patterns like *.pdf)'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing documents before uploading'
    )
    
    parser.add_argument(
        '--backend-url',
        default=DEFAULT_BACKEND_URL,
        help=f'Backend URL (default: {DEFAULT_BACKEND_URL})'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_colored("\n" + "="*60, Fore.CYAN)
    print_colored("  Financial Document Upload Utility", Fore.CYAN, Style.BRIGHT)
    print_colored("="*60, Fore.CYAN)
    
    # Check dependencies
    try:
        import requests
        import colorama
    except ImportError as e:
        print_colored(f"\n✗ Missing dependency: {e}", Fore.RED)
        print_colored("  Install with: pip install requests colorama", Fore.YELLOW)
        sys.exit(1)
    
    # Check backend health
    print_colored(f"\nChecking backend at {args.backend_url}...", Fore.CYAN)
    if not check_backend_health(args.backend_url):
        sys.exit(1)
    
    print_colored("✓ Backend is online", Fore.GREEN)
    
    # Expand file patterns
    file_paths = expand_file_patterns(args.files)
    
    if not file_paths:
        print_colored("\n✗ No files found matching the specified patterns", Fore.RED)
        print_colored("  Patterns searched:", Fore.YELLOW)
        for pattern in args.files:
            print_colored(f"    - {pattern}", Fore.YELLOW)
        sys.exit(1)
    
    # Validate all files
    print_colored(f"\nValidating {len(file_paths)} file(s)...", Fore.CYAN)
    valid_files = []
    
    for file_path in file_paths:
        is_valid, error_msg = validate_file(file_path)
        if is_valid:
            valid_files.append(file_path)
            print_colored(f"  ✓ {file_path.name}", Fore.GREEN)
        else:
            print_colored(f"  ✗ {file_path.name}: {error_msg}", Fore.RED)
    
    if not valid_files:
        print_colored("\n✗ No valid files to upload", Fore.RED)
        sys.exit(1)
    
    if len(valid_files) < len(file_paths):
        print_colored(f"\n⚠ Skipping {len(file_paths) - len(valid_files)} invalid file(s)", Fore.YELLOW)
    
    # Upload files
    success = upload_files(valid_files, args.backend_url, args.clear)
    
    if success:
        print_colored("\n✓ All done!\n", Fore.GREEN, Style.BRIGHT)
        sys.exit(0)
    else:
        print_colored("\n✗ Upload failed\n", Fore.RED, Style.BRIGHT)
        sys.exit(1)


if __name__ == "__main__":
    main()
