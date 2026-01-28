#!/usr/bin/env python3
"""
Terminal Chat Interface for RAG System

Interactive terminal-based chat interface for querying the Financial Document Q&A system.
Provides a REPL-style interface with commands for querying documents and managing the vector store.

Usage:
    python terminal_chat.py
    python terminal_chat.py --backend-url http://192.168.1.100:8001
"""

import argparse
import sys
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
from colorama import init, Fore, Style, Back

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configuration
DEFAULT_BACKEND_URL = "http://localhost:8000"
QUERY_TIMEOUT = 60  # 1 minute
HEALTH_TIMEOUT = 5
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.7


class ChatSession:
    """Manages chat session state and history"""
    
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.history: List[Dict[str, Any]] = []
        self.top_k = DEFAULT_TOP_K
        self.similarity_threshold = DEFAULT_SIMILARITY_THRESHOLD
    
    def add_to_history(self, query: str, response: Dict[str, Any]):
        """Add query and response to history"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response
        })


def print_colored(message: str, color: str = Fore.WHITE, style: str = Style.NORMAL):
    """Print colored message to console"""
    print(f"{style}{color}{message}{Style.RESET_ALL}")


def print_banner():
    """Display welcome banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║       📚 Financial Document Q&A - Terminal Chat 💬           ║
    ║                                                               ║
    ║       Powered by Ollama + FAISS + LangChain                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Fore.CYAN, Style.BRIGHT)
    print_colored("Type your questions about uploaded documents.", Fore.WHITE)
    print_colored("Type /help for available commands or /exit to quit.\n", Fore.WHITE)


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
        print_colored("\n  Make sure the backend is running:", Fore.YELLOW)
        print_colored("    cd backend", Fore.CYAN)
        print_colored("    uvicorn main:app --host 0.0.0.0 --port 8001\n", Fore.CYAN)
        return False
    except requests.exceptions.Timeout:
        print_colored("✗ Backend health check timed out", Fore.RED)
        return False
    except Exception as e:
        print_colored(f"✗ Error checking backend health: {e}", Fore.RED)
        return False


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source documents for display"""
    if not sources:
        return "  No sources available"
    
    output = []
    for idx, source in enumerate(sources, 1):
        relevance = source.get('relevance_score', 0) * 100
        metadata = source.get('metadata', {})
        content = source.get('content', '')
        
        output.append(f"\n  {Fore.YELLOW}Source {idx} [Relevance: {relevance:.0f}%]{Style.RESET_ALL}")
        
        # Display metadata
        file_name = metadata.get('source', 'Unknown')
        file_type = metadata.get('file_type', 'Unknown')
        
        output.append(f"    File: {file_name} ({file_type.upper()})")
        
        if 'page' in metadata:
            output.append(f"    Page: {metadata['page']}")
        elif 'rows' in metadata:
            output.append(f"    Rows: {metadata['rows']}")
        
        # Display preview
        preview = content[:150] + "..." if len(content) > 150 else content
        output.append(f"    Preview: \"{preview}\"")
    
    return "\n".join(output)


def display_stats(backend_url: str):
    """Fetch and display vector store statistics"""
    try:
        response = requests.get(f"{backend_url}/stats", timeout=HEALTH_TIMEOUT)
        
        if response.status_code == 200:
            stats = response.json()
            
            print_colored("\n" + "="*60, Fore.CYAN)
            print_colored("  Vector Store Statistics", Fore.CYAN, Style.BRIGHT)
            print_colored("="*60, Fore.CYAN)
            
            print_colored(f"\nCollection: {stats.get('collection_name', 'N/A')}", Fore.WHITE)
            print_colored(f"Documents: {stats.get('document_count', 0)}", Fore.WHITE)
            print_colored(f"Status: {stats.get('status', 'Unknown')}", Fore.WHITE)
            
            if stats.get('document_count', 0) == 0:
                print_colored("\n⚠ No documents in vector store.", Fore.YELLOW)
                print_colored("  Upload documents using: python scripts/upload_documents.py <files>", Fore.CYAN)
            
            print()
        else:
            print_colored(f"\n✗ Failed to fetch stats: HTTP {response.status_code}", Fore.RED)
            
    except Exception as e:
        print_colored(f"\n✗ Error fetching stats: {e}", Fore.RED)


def clear_documents(backend_url: str):
    """Clear all documents from vector store"""
    print_colored("\n⚠ This will delete all documents from the vector store.", Fore.YELLOW)
    confirm = input(f"{Fore.WHITE}Are you sure? (yes/no): {Style.RESET_ALL}").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print_colored("Cancelled.", Fore.CYAN)
        return
    
    try:
        response = requests.delete(f"{backend_url}/clear", timeout=HEALTH_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print_colored(f"\n✓ {result.get('message', 'Documents cleared successfully')}", Fore.GREEN)
        else:
            print_colored(f"\n✗ Failed to clear documents: HTTP {response.status_code}", Fore.RED)
            
    except Exception as e:
        print_colored(f"\n✗ Error clearing documents: {e}", Fore.RED)


def show_help():
    """Display help message"""
    help_text = f"""
{Fore.CYAN}{Style.BRIGHT}Available Commands:{Style.RESET_ALL}

  {Fore.GREEN}/help{Style.RESET_ALL}      - Show this help message
  {Fore.GREEN}/stats{Style.RESET_ALL}     - Display vector store statistics
  {Fore.GREEN}/clear{Style.RESET_ALL}     - Clear all documents from vector store
  {Fore.GREEN}/history{Style.RESET_ALL}   - Show conversation history
  {Fore.GREEN}/exit{Style.RESET_ALL}      - Exit the chat (or /quit, Ctrl+C)
  {Fore.GREEN}/quit{Style.RESET_ALL}      - Exit the chat

{Fore.CYAN}{Style.BRIGHT}Query Settings:{Style.RESET_ALL}

  Default top_k: {DEFAULT_TOP_K}
  Default similarity threshold: {DEFAULT_SIMILARITY_THRESHOLD}

{Fore.CYAN}{Style.BRIGHT}Tips:{Style.RESET_ALL}

  • Ask specific questions about the uploaded documents
  • Use natural language queries
  • Sources are ranked by relevance score
  • Check /stats to see how many documents are loaded
    """
    print(help_text)


def show_history(session: ChatSession):
    """Display conversation history"""
    if not session.history:
        print_colored("\nNo conversation history yet.", Fore.YELLOW)
        return
    
    print_colored("\n" + "="*60, Fore.CYAN)
    print_colored("  Conversation History", Fore.CYAN, Style.BRIGHT)
    print_colored("="*60, Fore.CYAN)
    
    for idx, entry in enumerate(session.history, 1):
        timestamp = entry['timestamp']
        query = entry['query']
        answer = entry['response'].get('answer', 'N/A')
        
        print_colored(f"\n[{idx}] {timestamp}", Fore.WHITE, Style.DIM)
        print_colored(f"Q: {query}", Fore.WHITE)
        print_colored(f"A: {answer[:100]}...", Fore.CYAN)
    
    print()


def send_query(session: ChatSession, query: str) -> bool:
    """
    Send query to backend and display response
    Returns: True if successful, False otherwise
    """
    try:
        payload = {
            "query": query,
            "top_k": session.top_k,
            "similarity_threshold": session.similarity_threshold
        }
        
        response = requests.post(
            f"{session.backend_url}/query",
            json=payload,
            timeout=QUERY_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display answer
            answer = result.get('answer', 'No answer generated')
            print_colored(f"\n{Fore.CYAN}{Style.BRIGHT}Answer:{Style.RESET_ALL}")
            print_colored(answer, Fore.CYAN)
            
            # Display sources
            sources = result.get('sources', [])
            if sources:
                print_colored(f"\n{Fore.YELLOW}{Style.BRIGHT}Sources:{Style.RESET_ALL}")
                print(format_sources(sources))
            else:
                print_colored("\nNo sources found above similarity threshold.", Fore.YELLOW)
            
            # Display metadata
            processing_time = result.get('processing_time', 0)
            tokens_used = result.get('tokens_used', 0)
            confidence = result.get('confidence', 0)
            
            print_colored(f"\n{Style.DIM}Processing time: {processing_time:.2f}s | "
                         f"Tokens: {tokens_used} | "
                         f"Confidence: {confidence:.0%}{Style.RESET_ALL}")
            
            # Add to history
            session.add_to_history(query, result)
            
            return True
            
        elif response.status_code == 404:
            print_colored("\n✗ No documents found in vector store.", Fore.RED)
            print_colored("  Upload documents first using: python scripts/upload_documents.py <files>", Fore.YELLOW)
            return False
            
        elif response.status_code == 500:
            print_colored(f"\n✗ Server error", Fore.RED)
            try:
                error_data = response.json()
                print_colored(f"  {error_data.get('message', 'Internal server error')}", Fore.YELLOW)
            except:
                pass
            return False
            
        else:
            print_colored(f"\n✗ Error: HTTP {response.status_code}", Fore.RED)
            return False
            
    except requests.exceptions.Timeout:
        print_colored("\n✗ Query timed out", Fore.RED)
        print_colored("  Try a simpler query or increase timeout", Fore.YELLOW)
        return False
        
    except requests.exceptions.ConnectionError:
        print_colored("\n✗ Connection error - backend may be down", Fore.RED)
        return False
        
    except Exception as e:
        print_colored(f"\n✗ Error: {e}", Fore.RED)
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive terminal chat for Financial Document Q&A",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--backend-url',
        default=DEFAULT_BACKEND_URL,
        help=f'Backend URL (default: {DEFAULT_BACKEND_URL})'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    try:
        import requests
        import colorama
    except ImportError as e:
        print_colored(f"✗ Missing dependency: {e}", Fore.RED)
        print_colored("  Install with: pip install requests colorama", Fore.YELLOW)
        sys.exit(1)
    
    # Check backend health
    print_colored(f"Connecting to backend at {args.backend_url}...", Fore.CYAN)
    if not check_backend_health(args.backend_url):
        sys.exit(1)
    
    print_colored("✓ Connected to backend\n", Fore.GREEN)
    
    # Create session
    session = ChatSession(args.backend_url)
    
    # Show initial stats
    display_stats(args.backend_url)
    
    # Main chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = input(f"\n{Fore.WHITE}{Style.BRIGHT}You: {Style.RESET_ALL}").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command in ['/exit', '/quit']:
                    print_colored("\nGoodbye! 👋\n", Fore.CYAN)
                    break
                    
                elif command == '/help':
                    show_help()
                    
                elif command == '/stats':
                    display_stats(args.backend_url)
                    
                elif command == '/clear':
                    clear_documents(args.backend_url)
                    
                elif command == '/history':
                    show_history(session)
                    
                else:
                    print_colored(f"\n✗ Unknown command: {user_input}", Fore.RED)
                    print_colored("  Type /help for available commands", Fore.YELLOW)
            
            else:
                # Send query
                send_query(session, user_input)
    
    except KeyboardInterrupt:
        print_colored("\n\nGoodbye! 👋\n", Fore.CYAN)
        sys.exit(0)


if __name__ == "__main__":
    main()
