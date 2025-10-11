import os
import json
import logging
import re

logger = logging.getLogger(__name__)


def clean_response(response: str, response_type: str = "text") -> str:
    """Clean LLM response by removing markdown formatting and extracting content."""
    if response_type == "csv":
        # Extract CSV content from markdown code blocks
        csv_pattern = r'```(?:csv)?\s*\n(.*?)\n```'
        matches = re.findall(csv_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()
    
    elif response_type == "json":
        # Extract JSON content from markdown code blocks
        json_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()
    
    return response.strip()


def alphabetize_csv_text(csv_text: str) -> str:
    """Alphabetize CSV entries while preserving the header."""
    lines = csv_text.strip().split('\n')
    if len(lines) <= 1:
        return csv_text
    
    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip()]
    
    # Sort data lines alphabetically
    data_lines.sort()
    
    return '\n'.join([header] + data_lines)


def get_csv_text_n_entries(csv_text: str) -> int:
    """Count the number of data entries in CSV text (excluding header)."""
    lines = csv_text.strip().split('\n')
    # Count non-empty lines excluding the header
    return len([line for line in lines[1:] if line.strip()]) if len(lines) > 1 else 0


def load_required_files(memory_dir: str, required_files: dict) -> dict:
    """Load required files from memory directory."""
    files = {}
    
    for key, filename in required_files.items():
        # Try to find the file in the appropriate subdirectory
        file_path = os.path.join(memory_dir, key, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                files[key] = f.read()
            logger.info(f"Loaded {key}: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    return files


def save_memory(content: str, memory_dir: str, filename: str, metadata: dict):
    """Save content and metadata to memory directory."""
    os.makedirs(memory_dir, exist_ok=True)
    
    # Save content
    content_path = os.path.join(memory_dir, filename)
    with open(content_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Save metadata
    metadata_path = os.path.join(memory_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {filename} and metadata to {memory_dir}")