import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from dotenv import load_dotenv
from main import Config, CodeAnalyzer, QuestionAnsweringService, CodeQAApp, initialize_app, QueryModel, QueryRequest


@pytest.fixture
def test_config(key=None):
    """Provide test configuration."""

    if key is None:
        if not load_dotenv():
            #".env" not in os.listdir():
            load_dotenv("../.env")
        else:
            load_dotenv()

        decrypted_key = os.getenv("OPENAI_API_KEY", None)
    else:
        pass
        '''if isinstance(key,str):
            decrypted_key = key
        else:
            decrypted_key = ""
            '''

    config = Config(
        key="test-key.gpg",
        model="gpt-4",
        embedding="MiniLM-L6",
        repo="test_repo"
    )
    config.key=decrypted_key


    return config


@pytest.fixture
def test_environment(tmp_path):
    """
    Create a test environment with sample files.
    """
    test_file = tmp_path / "test.py"
    test_content = """
def test_function():
    '''Test function docstring'''
    return True

class TestClass:
    '''Test class docstring'''
    def method(self):
        pass
"""
    test_file.write_text(test_content)
    return tmp_path


'''
@pytest.fixture
def mock_openai(real_key=False):
    """Set up mock for OpenAI API interactions."""
    with patch('langchain_openai.ChatOpenAI', autospec=True) as mock_chat:
        mock_instance = mock_chat.return_value

        # Create an async mock for the ainvoke method
        async def mock_ainvoke(*args, **kwargs):
            return "Mocked response"

        mock_instance.ainvoke = mock_ainvoke
        mock_instance.invoke = mock_ainvoke

        if real_key:
            load_dotenv()
            decrypted_key = os.getenv("OPENAI_API_KEY", None)

            yield mock_chat

        else:
            os.environ["OPENAI_API_KEY"] = "test-key"

            yield mock_chat

            # Clean up
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
'''


class TestCodeAnalyzer:
    def test_file_validation(self, test_environment):
        """Test file validation logic."""
        analyzer = CodeAnalyzer(test_environment)

        # Test valid file
        python_file = Path(test_environment) / "test.py"
        assert analyzer._is_valid_file(python_file)

        # Test non-existent file
        missing_file = Path(test_environment) / "nonexistent.py"
        assert not analyzer._is_valid_file(missing_file)

    def test_python_parsing(self, test_environment):
        analyzer = CodeAnalyzer(test_environment)
        python_file = Path(test_environment) / "test.py"
        chunks = analyzer._parse_python_file(python_file.read_text(), str(python_file))

        assert len(chunks) == 3  # Should find standalone function, class, and class method
        assert any(c['type'] == 'FunctionDef' and c['name'] == 'test_function' for c in chunks)
        assert any(c['type'] == 'ClassDef' and c['name'] == 'TestClass' for c in chunks)
        assert any(c['type'] == 'FunctionDef' and c['name'] == 'method' for c in chunks)


class TestQuestionAnswering:
    @pytest.mark.asyncio
    async def test_query_handling(self, test_environment, test_config):
        """Test the query handling functionality."""
        # Setup test file with content
        test_file = Path(test_environment) / "test.py"
        test_file.write_text("def example(): return 'test'")

        # Set up test configuration
        test_config.repo = str(test_environment)

        try:
            # Initialize application with test configuration
            app = initialize_app(test_config)

            # Test query processing
            request = QueryRequest(query="What does the example function do?")
            response = await app.analyze_code(request)

            # Verify response structure
            assert response is not None
            assert isinstance(response.answer, str)
            assert len(response.answer) > 0
            assert isinstance(response.context, list)

        except Exception as e:
            pytest.fail(f"Test failed with exception: {str(e)}")

    '''
    @pytest.mark.asyncio
    async def test_query_handling_MOCK(self, test_environment, mock_openai, test_config):
        """Test the query handling functionality."""
        # Setup test file with content
        test_file = Path(test_environment) / "test.py"
        test_file.write_text("def example(): return 'test'")

        # Set up test configuration
        test_config.repo = str(test_environment)

        try:
            # Create a test instance of QuestionAnsweringService with the mock
            qa_service = QuestionAnsweringService(None, llm=mock_openai.return_value)

            # Initialize application with test configuration and inject the mock service
            app = initialize_app(test_config)
            app.qa_service = qa_service

            # Test query processing
            request = QueryRequest(query="What does the example function do?")
            response = await app.analyze_code(request)

            # Verify response structure and content
            assert response is not None
            assert response.answer == "Mocked response"
            assert isinstance(response.context, list)

        except Exception as e:
            pytest.fail(f"Test failed with exception: {str(e)}")
            '''


    def test_error_handling(self, test_environment, test_config):
        # Test with invalid repository path
        with pytest.raises(ValueError):
            CodeQAApp(repo_path="/nonexistent/path")


if __name__ == '__main__':
    pytest.main(['-v'])