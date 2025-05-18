import os
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('code_qa_app.log'),
            logging.StreamHandler()
        ]
    )


def get_file_extension_types() -> Dict[str, List[str]]:
    """
    Returns a dictionary mapping file types to their extensions.
    This is a utility function to centralize extension management.
    """
    return {
        # Code files
        'python': ['.py', '.pyx', '.pyw', '.ipynb'],
        'cpp': ['.cpp', '.hpp', '.h', '.cc', '.cxx', '.c'],
        'java': ['.java', '.jar'],
        'javascript': ['.js', '.jsx', '.ts', '.tsx'],
        'rust': ['.rs'],
        'go': ['.go'],
        # Documentation
        'markdown': ['.md', '.markdown'],
        'text': ['.txt', '.log'],
        'json': ['.json'],
        'yaml': ['.yml', '.yaml'],
        # PDF support
        'pdf': ['.pdf'],
        # Binary files that should be ignored
        'binary': ['.ico', '.ttf', '.eot', '.woff', '.woff2', '.pdn']
    }


def get_root_path() -> str:
    """
    Determine the root path of the application.
    """
    rootpath = os.path.dirname(os.path.abspath(__file__))
    if rootpath.endswith('src/'):
        rootpath = rootpath[:-4]
    elif rootpath.endswith('src'):
        rootpath = rootpath[:-3]
    return rootpath