import ast
import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


def is_valid_file(file_path: Path) -> bool:
    try:
        if not file_path.is_file():
            return False
        binary_extensions = {'.ico', '.ttf', '.eot', '.woff', '.woff2', '.pdn'}
        if file_path.suffix.lower() in binary_extensions or file_path.name == '.DS_Store':
            return False
        # For non-PDF files we attempt to read a small portion to ensure text content
        if file_path.suffix.lower() != '.pdf':
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1)
        return True
    except Exception as e:
        logger.debug(f"Skipping non-text file {file_path}: {e}")
        return False


def parse_python_file(content: str, filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    try:
        module = ast.parse(content)
        for node in ast.walk(module):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                chunk = {
                    'content': ast.get_source_segment(content, node) or '',
                    'type': type(node).__name__,
                    'name': getattr(node, 'name', 'Module'),
                    'filepath': filepath,
                    'line_number': getattr(node, 'lineno', 0),
                    'language': 'python'
                }
                if docstring := ast.get_docstring(node):
                    chunk['docstring'] = docstring
                chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error parsing Python file {filepath}: {e}")
    return chunks


def parse_cpp_file(content: str, filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    patterns = [
        (r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{(?:[^{}]|{[^{}]*})*\}', 'ClassDef'),
        (r'(?:virtual\s+)?(?:\w+(?:::\w+)*\s+)+(\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:override\s*)?(?:=\s*0\s*)?(?:noexcept\s*)?\s*\{(?:[^{}]|{[^{}]*})*\}', 'FunctionDef')
    ]
    for pattern, chunk_type in patterns:
        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            chunk = {
                'content': match.group(0),
                'type': chunk_type,
                'name': match.group(1),
                'filepath': filepath,
                'line_number': content[:match.start()].count('\n') + 1,
                'language': 'cpp'
            }
            chunks.append(chunk)
    return chunks


def parse_markdown_file(content: str, filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    sections = content.split('\n#')
    for section in sections:
        if section.strip():
            chunks.append({
                'content': section.strip(),
                'type': 'Documentation',
                'name': section.split('\n')[0].strip('# '),
                'filepath': filepath,
                'format': 'Markdown'
            })
    return chunks


def parse_generic_file(content: str, filepath: str) -> List[Dict[str, Any]]:
    chunks = []
    lines = content.split('\n')
    chunk_size = 50
    for i in range(0, len(lines), chunk_size):
        chunk_content = '\n'.join(lines[i:i + chunk_size])
        if chunk_content.strip():
            chunks.append({
                'content': chunk_content,
                'type': 'Generic',
                'name': f'Chunk {i // chunk_size + 1}',
                'filepath': filepath,
                'line_number': i + 1
            })
    return chunks


def parse_pdf_file(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from a PDF file and split it into chunks."""
    chunks = []
    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        # Split the extracted text into manageable chunks (e.g., 50 lines per chunk)
        lines = full_text.split('\n')
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk_text = '\n'.join(lines[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'content': chunk_text,
                    'type': 'PDF',
                    'name': f'PDF Chunk {i // chunk_size + 1}',
                    'filepath': file_path,
                    'line_number': i + 1
                })
    except Exception as e:
        logger.error(f"Error parsing PDF file {file_path}: {e}")
    return chunks