import os
import subprocess
import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv


def parse_arguments():
    parser = argparse.ArgumentParser(description="Database and API for Code Q&A")
    parser.add_argument(
        "--key",
        help="name of key to use. key should be placed in \".keys\" directory ; default = \"openai-steve-sa-key.gpg\"",
        default="openai-steve-sa.key",
        required=False
    )
    parser.add_argument(
        "--local-model",
        help="local model URL; default = \"http://localhost:1234/v1/chat/completions\"",
        default="http://localhost:1234/v1/chat/completions",
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


def setup_environment():
    args = parse_arguments()
    global OPENAI_API_KEY, EMBEDDING_MODEL, REPO_PATH, LLM, LOCAL_MODEL

    load_dotenv()
    rootpath = os.path.dirname(os.path.abspath(__file__))
    if rootpath.endswith('src/'):
        rootpath = rootpath[:-4]
    elif rootpath.endswith('src'):
        rootpath = rootpath[:-3]

    KEYPATH = os.path.join(rootpath, ".keys", args.key)
    
    # Set local model URL if provided
    LOCAL_MODEL = args.local_model if args.local_model else None

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

    if decrypted_key is None and not LOCAL_MODEL:
        raise Exception("No OpenAI API key found in .env or GPG file and no local model specified")
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


class Config:
    PRODUCTION_DEFAULTS = {
        'key': "openai-steve-sa.key",
        'model': "gpt-4o",
        'embedding': "MiniLM-L6",
        'repo': "./TEST/grip-no-tests",
        'local_model': None
    }
    TEST_DEFAULTS = {
        'key': "openai-steve-sa.key",
        'model': "gpt-4",
        'embedding': "MiniLM-L6",
        'repo': "./TEST/test-repo",
        'local_model': None
    }

    def __init__(self, environment='production', **kwargs):
        self.environment = environment
        defaults = self.TEST_DEFAULTS if environment == 'test' else self.PRODUCTION_DEFAULTS
        self.key = kwargs.get('key', defaults['key'])
        self.model = kwargs.get('model', defaults['model'])
        self.embedding = kwargs.get('embedding', defaults['embedding'])
        self.repo = kwargs.get('repo', defaults['repo'])
        self.local_model = kwargs.get('local_model', defaults['local_model'])
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


def initialize_app(config: Config):
    # Import here to avoid circular import
    from .code_analyzer import CodeQAApp
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('code_qa_app.log'), logging.StreamHandler()]
    )
    return CodeQAApp(repo_path=config.repo, embedding_model=config.embedding, local_model=config.local_model)