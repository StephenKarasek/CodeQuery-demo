# Overview
REST API that answers questions about a local repo (Python, C++, etc.) using a combination of retrieval and language models.

## Usage
* Given the root path of a project:
  - *Accept questions in natural language*
  - *Answer them as text (with code snippets)*
* The questions might include things like:
  - *What does class X do?*
  - *How is service Y implemented?*
  - *How does method X use parameter Y?*


### Startup
```bash
python -m src.main --repo /path/to/codebase/dir
```
This will start up the tool, process the codebase into a vector database, and confirm when it is ready for queries.

#### Input
* The API accepts a JSON object with the following fields:
  - `query`: the question to answer

#### Output
* The API returns a JSON object with the following fields:
  - `answer`: the answer to the question
  - `context`: code segments used to generate the answer

## LLM
You will need an OpenAI key, or run an LLM locally

### API Key
You can use a GPG file, or add the key value to environment variables
* GPG File
  - Obtain from OpenAI
  - place in "/.keys"

* Environment Variable
  - Store in ENV file (environment variable)
  - OPENAI_API_KEY="key-value-goes-here"



### Run Locally
* Run LLM locally
  - Ollama
  - LMStudio

#### LM Studio
* Download https://lmstudio.ai/
* Retrieve model (example uses Llama-3.2):
  bartowski/Llama-3.2-3B-Instruct-GGUF
* Load model in "Local Server"
* Use curl command to retrieve information

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{ 
    "model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "messages": [ 
      { "role": "system", "content": "Always answer in rhymes." },
      { "role": "user", "content": "Introduce yourself." }
    ], 
    "temperature": 0.7, 
    "max_tokens": -1,
    "stream": true
}'
```

* TODO: Implement Chat

################################################################
# Examples

### pymc-main
API key
```bash
python -m src.main --repo TEST/pymc-main --key /path/to/key
```
Local Model
```bash
python -m src.main --repo TEST/pymc-main --local-model http://localhost:1234
```
Query
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "How does the Metropolis-Hastings algorithm work?"}'
```
RESULT: success; returns answer & explanation


### scikit-learn-main
API key
```bash
python -m src.main --repo TEST/scikit-learn-main --key /path/to/key
```
Local Model
```bash
python -m src.main --repo TEST/scikit-learn-main --local-model http://localhost:1234
```
Query
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "How does the RandomForestClassifier work?"}'
```
 RESULT: MIXED; not enough information to answer, but provides contact info from the repo owners


### grip-no-tests
API key
```bash
python -m src.main --repo TEST/grip-no-tests --key /path/to/key
```
Local Model
```bash
python -m src.main --repo TEST/grip-no-tests --local-model http://localhost:1234
```
Query
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What does the create_app function do? what file is it present in?"}'
```
RESULT: success; returns answer & explanation


### OpenFOAM-dev-master
API key
```bash
python -m src.main --repo TEST/OpenFOAM-dev-master --key /path/to/key
```
Local Model
```bash
python -m src.main --repo TEST/OpenFOAM-dev-master --local-model http://localhost:1234
```
Query
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "How could I create a k-epsilon turbulent flow simulation?"}'
```
RESULT: success; returns answer & explanation



################################################################
# Components


### RAG (retrieval augmented generation)
* define & implement:
  - chunking code
  - storage
  - indexing
  - retrieval

### LLMs
* API
  * GPT-4o (working!)
  * (TODO: Anthropic Claude, Mistral, …)
* Locally
  * use LMStudio, Ollama, HuggingFace, ...
  * Tested on Llama & Mistral



### Input Size
* Support large repos
  * (i.e. those that could not fit in a model's context window)



### Unit Tests
Run tests with:
```
pytest -v --cov=src
python -m pytest src/test_code_qa.py -v
```
### CodeAnalyzer
* File validation logic
* Python file parsing
* C++ file parsing
* Vector index creation and searching
* Support for different file types
####
### QuestionAnsweringService
* Answer generation with mock context
* Error handling
####
### CodeQAApp
* Application initialization 
* API endpoint functionality
* Response format validation


## Project Organization

The project is organized into multiple modules:

- **config.py**: Configuration and environment setup
- **code_analyzer.py**: Core functionality for analyzing code repositories
- **parsers.py**: File parsing for different file types
- **qa_service.py**: Question answering using LLMs
- **utils.py**: Utility functions and helpers
- **main.py**: Application entry point

