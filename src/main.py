import os
import subprocess
import logging
import traceback
import argparse
from typing import List, Dict, Any
import ast
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# New import for PDF processing
from PyPDF2 import PdfReader


def parse_arguments():
    parser = argparse.ArgumentParser(description="Database and API for Code Q&A")
    parser.add_argument(
        "--key",
        help="name of key to use. key should be placed in \".keys\" directory ; default = \"openai-steve-sa-key.gpg\"",
        default="openai-steve-sa.key",
        required=False
    )
    parser.add_argument(
        "--model",
        help="name of LLM to use. Options include: gpt-4o, davinci, CUSTOM ; default = \"gpt-4o\"",
        default="gpt-4o",
        required=False
    )
    parser.add_argument(
        "--embedding",
        help="name of embedding model to use. Options include: MiniLM-L6, paragraph-mpnet-base-v2",
        default="MiniLM-L6",
        required=False
    )
    parser.add_argument(
        "--repo",
        help="path to repository directory; default = \"./TEST/grip-no-tests\"",
        default="./TEST/grip-no-tests",
        required=False
    )
    return parser.parse_args()


def setup_environment():
    args = parse_arguments()
    global OPENAI_API_KEY, EMBEDDING_MODEL, REPO_PATH, LLM

    load_dotenv()
    rootpath = os.path.dirname(os.path.abspath(__file__))
    if rootpath.endswith('src/'):
        rootpath = rootpath[:-4]
    elif rootpath.endswith('src'):
        rootpath = rootpath[:-3]

    KEYPATH = os.path.join(rootpath, ".keys", args.key)

    load_dotenv()
    GPG_PASSPHRASE = os.getenv("GPG_PASSPHRASE")
    try:
        print("Loading OpenAI Key from .env")
        decrypted_key = os.getenv("OPENAI_API_KEY", None)
    except Exception as e:
        print(e)
        print("Loading OpenAI Key from GPG file")
        try:
            decrypted_key = get_openai_key_from_gpg(KEYPATH, GPG_PASSPHRASE)
        except Exception as e:
            print(e)
            decrypted_key = None

    if decrypted_key is None:
        raise Exception("No OpenAI API key found in .env or GPG file")
    else:
        OPENAI_API_KEY = decrypted_key

    if args.model.lower() == "gpt-4o":
        LLM = "text-gpt-4o"
    elif args.model.lower() == "CUSTOM":
        LLM = ""
    else:
        LLM = "text-gpt-4o"

    if args.embedding.lower() == "MiniLM-L6".lower():
        EMBEDDING_MODEL = "MiniLM-L6"
    elif args.embedding.lower() == "MiniLM-L12".lower():
        EMBEDDING_MODEL = "MiniLM-L12"
    elif args.embedding.lower() == "MPNet-base".lower():
        EMBEDDING_MODEL = "MPNet-base"
    elif args.embedding.lower() == "CodeBERT-base".lower():
        EMBEDDING_MODEL = "CodeBERT-base"
    elif args.embedding.lower() == "GraphCodeBERT-base".lower():
        EMBEDDING_MODEL = "GraphCodeBERT-base"
    elif args.embedding.lower() == "CodeBERT-base-tl".lower():
        EMBEDDING_MODEL = "CodeBERT-base-tl"
    elif args.embedding.lower() == "MultiCodeBERT-base".lower():
        EMBEDDING_MODEL = "MultiCodeBERT-base"
    else:
        EMBEDDING_MODEL = "MiniLM-L6"

    if os.path.exists(args.repo):
        REPO_PATH = args.repo
    else:
        print("Using default test repository: grip-no-tests")
        REPO_PATH = os.path.join(rootpath, "TEST", "grip-no-tests")


def get_openai_key_from_gpg(gpg_file_path: str, gpg_passphrase: str) -> str:
    result = subprocess.run(
        [
            "gpg",
            "--quiet",
            "--batch",
            "--yes",
            "--pinentry-mode=loopback",
            "--passphrase", gpg_passphrase,
            "--decrypt", gpg_file_path
        ],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise Exception(f"Error decrypting {gpg_file_path}: {result.stderr}")
    return result.stdout.strip()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_qa_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Query Model
class QueryModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


# Code Analyzer with PDF support
class CodeAnalyzer:
    # Added 'pdf' key for PDF files
    embedding_models = {
        'MiniLM-L6': 'sentence-transformers/all-MiniLM-L6-v2',
        'MiniLM-L12': 'sentence-transformers/all-MiniLM-L12-v2',
        'MPNet-base': 'sentence-transformers/all-mpnet-base-v2',
        'CodeBERT-base': 'microsoft/codebert-base',
        'GraphCodeBERT-base': 'microsoft/graphcodebert-base',
        'CodeBERT-base-tl': 'microsoft/codebert-base-transfer-learning',
        'MultiCodeBERT-base': 'microsoft/multilingual-codebert-base'
    }

    supported_extensions = {
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
        # PDF support added here
        'pdf': ['.pdf']
    }

    def __init__(self, repo_path: str, embedding_model: str = 'default'):
        self.repo_path = repo_path
        if embedding_model not in self.embedding_models:
            logger.warning(f"Invalid embedding model: {embedding_model}. Using default.")
            embedding_model = self.embedding_models['MiniLM-L6']
        else:
            embedding_model = self.embedding_models[embedding_model]
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    def extract_code_chunks(self) -> List[Dict[str, Any]]:
        chunks = []
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                file_path = Path(os.path.join(root, file))
                ext_ = file_path.suffix.lower()
                try:
                    if not self._is_valid_file(file_path):
                        continue

                    # Process code, markdown, text, and now PDF files
                    if ext_ in self.supported_extensions.get('python', []) or \
                       ext_ in self.supported_extensions.get('cpp', []) or \
                       ext_ in self.supported_extensions.get('java', []) or \
                       ext_ in self.supported_extensions.get('javascript', []) or \
                       ext_ in self.supported_extensions.get('rust', []) or \
                       ext_ in self.supported_extensions.get('go', []) or \
                       ext_ in self.supported_extensions.get('markdown', []) or \
                       ext_ in self.supported_extensions.get('text', []) or \
                       ext_ in self.supported_extensions.get('json', []) or \
                       ext_ in self.supported_extensions.get('yaml', []):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if ext_ in self.supported_extensions['python']:
                            new_chunks = self._parse_python_file(content, str(file_path))
                        elif ext_ in self.supported_extensions['cpp']:
                            new_chunks = self._parse_cpp_file(content, str(file_path))
                        elif ext_ in self.supported_extensions['markdown']:
                            new_chunks = self._parse_markdown_file(content, str(file_path))
                        else:
                            new_chunks = self._parse_generic_file(content, str(file_path))
                        chunks.extend(new_chunks)

                    # Process PDF files using the dedicated parser
                    elif ext_ in self.supported_extensions.get('pdf', []):
                        new_chunks = self._parse_pdf_file(str(file_path))
                        chunks.extend(new_chunks)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

        if not chunks:
            logger.warning(f"No valid chunks found in {self.repo_path}")
            raise ValueError("No valid content could be extracted from the repository")
        return chunks

    def _is_valid_file(self, file_path: Path) -> bool:
        try:
            if not file_path.is_file():
                return False
            binary_extensions = {'.ico', '.ttf', '.eot', '.woff', '.woff2', '.pdn'}
            if file_path.suffix.lower() in binary_extensions or file_path.name == '.DS_Store':
                return False
            # For non-PDF files we attempt to read a small portion to ensure text content
            if file_path.suffix.lower() not in self.supported_extensions.get('pdf', []):
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(1)
            return True
        except Exception as e:
            logger.debug(f"Skipping non-text file {file_path}: {e}")
            return False

    def _parse_python_file(self, content: str, filepath: str) -> List[Dict[str, Any]]:
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

    def _parse_cpp_file(self, content: str, filepath: str) -> List[Dict[str, Any]]:
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

    def _parse_markdown_file(self, content: str, filepath: str) -> List[Dict[str, Any]]:
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

    def _parse_generic_file(self, content: str, filepath: str) -> List[Dict[str, Any]]:
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

    def _parse_pdf_file(self, file_path: str) -> List[Dict[str, Any]]:
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

    def create_vector_index(self, chunks: List[Dict[str, Any]]) -> None:
        texts = []
        for chunk in chunks:
            chunk_text = [
                f"File: {chunk['filepath']}",
                f"Type: {chunk['type']}",
                f"Name: {chunk.get('name', 'N/A')}"
            ]
            if 'language' in chunk:
                chunk_text.append(f"Language: {chunk['language']}")
            if 'docstring' in chunk:
                chunk_text.append(f"Documentation: {chunk['docstring']}")
            chunk_text.append(f"Content:\n{chunk['content']}")
            texts.append('\n'.join(chunk_text))
        self.vectorstore = FAISS.from_texts(texts, self.embedding_model)


# Question Answering Service with fixed output
class QuestionAnsweringService:
    def __init__(self, analyzer: CodeAnalyzer, temp=0.2):
        self.analyzer = analyzer
        self.llm = ChatOpenAI(model='gpt-4o', temperature=temp)

    async def generate_answer(self, query: str, context: List[str]) -> str:
        prompt = PromptTemplate(
            template="""
You are an expert code analyst. Provide a detailed and precise answer based strictly on the provided code context.
If possible, provide a code snippet and short example to show the user how to use it.

Guidelines:
- If the answer cannot be definitively found, state: "I cannot find a clear answer in the provided code."
- Be specific and reference exact code segments if possible.
- Focus on explaining the implementation, purpose, and key characteristics.

Question: {query}

Code Context:
{context}

Detailed Technical Analysis:""",
            input_variables=['query', 'context']
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        try:
            answer = await chain.arun(query=query, context='\n\n'.join(context))
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"An error occurred while generating the answer: {str(e)}"


# FastAPI Models for Request and Response
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The question about the code")


class QueryResponse(BaseModel):
    answer: str
    context: List[str]


# Code QA App
class CodeQAApp:
    k = 3

    def __init__(self, repo_path: str, embedding_model: str = 'default'):
        self.analyzer = CodeAnalyzer(repo_path, embedding_model)
        self.app = FastAPI()
        self.qa_service = None
        logger.info("Starting code analysis...")
        chunks = self.analyzer.extract_code_chunks()
        logger.info(f"Extracted {len(chunks)} code chunks")
        self.analyzer.create_vector_index(chunks)
        logger.info("Vector index created successfully")
        self._setup_routes()

    async def analyze_code(self, request: QueryRequest) -> QueryResponse:
        try:
            relevant_docs = self.analyzer.vectorstore.similarity_search(request.query, k=self.k)
            relevant_context = [doc.page_content for doc in relevant_docs]
            logger.info(f"Found {len(relevant_context)} relevant code segments")
            qa_service = QuestionAnsweringService(self.analyzer)
            answer = await qa_service.generate_answer(request.query, relevant_context)
            return QueryResponse(answer=answer, context=relevant_context)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _setup_routes(self):
        @self.app.post("/ask", response_model=QueryResponse)
        async def handle_request(request: QueryRequest):
            return await self.analyze_code(request)

    def run(self, host: str = '0.0.0.0', port: int = 8000):
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Configuration classes (unchanged)
class Config:
    PRODUCTION_DEFAULTS = {
        'key': "openai-steve-sa.key",
        'model': "gpt-4o",
        'embedding': "MiniLM-L6",
        'repo': "./TEST/grip-no-tests"
    }
    TEST_DEFAULTS = {
        'key': "openai-steve-sa.key",
        'model': "gpt-4",
        'embedding': "MiniLM-L6",
        'repo': "./TEST/test-repo"
    }

    def __init__(self, environment='production', **kwargs):
        self.environment = environment
        defaults = self.TEST_DEFAULTS if environment == 'test' else self.PRODUCTION_DEFAULTS
        self.key = kwargs.get('key', defaults['key'])
        self.model = kwargs.get('model', defaults['model'])
        self.embedding = kwargs.get('embedding', defaults['embedding'])
        self.repo = kwargs.get('repo', defaults['repo'])
        self._initialize_paths()

    def get_openai_key(self) -> str:
        try:
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return key
            gpg_path = os.path.join(self.rootpath, ".keys", self.key)
            passphrase = os.getenv("GPG_PASSPHRASE")
            if not passphrase:
                raise ValueError("GPG_PASSPHRASE not found in environment")
            return get_openai_key_from_gpg(gpg_path, passphrase)
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve OpenAI API key: {e}")

    def _initialize_paths(self):
        self.rootpath = os.path.dirname(os.path.abspath(__file__))
        if self.rootpath.endswith('src/'):
            self.rootpath = self.rootpath[:-4]
        elif self.rootpath.endswith('src'):
            self.rootpath = self.rootpath[:-3]
        if not os.path.isabs(self.repo):
            self.repo = os.path.join(self.rootpath, self.repo)

    @classmethod
    def from_command_line(cls):
        args = parse_arguments()
        return cls(**vars(args))

    @classmethod
    def create_test_config(cls, **override_kwargs):
        return cls(environment='test', **override_kwargs)

    def __str__(self):
        return f"Config({self.environment})[key={self.key}, model={self.model}, embedding={self.embedding}, repo={self.repo}]"


def initialize_app(config: Config) -> CodeQAApp:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('code_qa_app.log'), logging.StreamHandler()]
    )
    return CodeQAApp(repo_path=config.repo, embedding_model=config.embedding)


def main():
    try:
        config = Config.from_command_line()
        openai_key = config.get_openai_key()
        os.environ["OPENAI_API_KEY"] = openai_key
        qa_app = initialize_app(config)
        qa_app.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    setup_environment()
    main()

'''
FastAPI application; serves as Q&A system for code-related queries:
(1) Code analyzer extracts code chunks (including text from PDF files) from a specified repository path.
(2) Creates a vector index of the extracted code chunks.
(3) Responds to queries based on the analyzed code context.
'''





