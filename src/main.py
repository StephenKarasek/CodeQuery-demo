import os
import logging
import traceback

from .utils import setup_logging
from .config import Config, setup_environment
from .code_analyzer import CodeQAApp

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    try:
        config = Config.from_command_line()
        openai_key = config.get_openai_key()
        os.environ["OPENAI_API_KEY"] = openai_key
        qa_app = CodeQAApp(repo_path=config.repo, embedding_model=config.embedding, local_model=config.local_model)
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