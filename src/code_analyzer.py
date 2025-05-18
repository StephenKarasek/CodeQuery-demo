import os
import logging
import traceback
from typing import List, Dict, Any
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .parsers import (
    is_valid_file, 
    parse_python_file, 
    parse_cpp_file, 
    parse_markdown_file, 
    parse_generic_file,
    parse_pdf_file
)
from .qa_service import QuestionAnsweringService, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


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

    def __init__(self, repo_path: str, embedding_model: str = 'default', local_model: str = None):
        self.repo_path = repo_path
        self.local_model = local_model
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
                    if not is_valid_file(file_path):
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
                            new_chunks = parse_python_file(content, str(file_path))
                        elif ext_ in self.supported_extensions['cpp']:
                            new_chunks = parse_cpp_file(content, str(file_path))
                        elif ext_ in self.supported_extensions['markdown']:
                            new_chunks = parse_markdown_file(content, str(file_path))
                        else:
                            new_chunks = parse_generic_file(content, str(file_path))
                        chunks.extend(new_chunks)

                    # Process PDF files using the dedicated parser
                    elif ext_ in self.supported_extensions.get('pdf', []):
                        new_chunks = parse_pdf_file(str(file_path))
                        chunks.extend(new_chunks)

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue

        if not chunks:
            logger.warning(f"No valid chunks found in {self.repo_path}")
            raise ValueError("No valid content could be extracted from the repository")
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


class CodeQAApp:
    k = 3

    def __init__(self, repo_path: str, embedding_model: str = 'default', local_model: str = None):
        self.analyzer = CodeAnalyzer(repo_path, embedding_model, local_model)
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
            logger.info(f"Processing query: {request.query}")
            relevant_docs = self.analyzer.vectorstore.similarity_search(request.query, k=self.k)
            relevant_context = [doc.page_content for doc in relevant_docs]
            logger.info(f"Found {len(relevant_context)} relevant code segments")
            
            qa_service = QuestionAnsweringService(self.analyzer)
            logger.info("Created QuestionAnsweringService instance")
            
            answer = await qa_service.generate_answer(request.query, relevant_context)
            logger.info("Generated answer successfully")
            
            return QueryResponse(answer=answer, context=relevant_context)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def _setup_routes(self):
        @self.app.post("/ask", response_model=QueryResponse)
        async def handle_request(request: QueryRequest):
            logger.info("Received request at /ask endpoint")
            response = await self.analyze_code(request)
            logger.info("Sending response back to client")
            return response

    def run(self, host: str = '0.0.0.0', port: int = 8000):
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")