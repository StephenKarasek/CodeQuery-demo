import logging
import traceback
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)


class QueryModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="The question about the code")


class QueryResponse(BaseModel):
    answer: str
    context: List[str]


class QuestionAnsweringService:
    def __init__(self, analyzer, temp=0.2):
        self.analyzer = analyzer
        if hasattr(analyzer, 'local_model') and analyzer.local_model:
            # Use local model
            base_url = analyzer.local_model
            # Remove /v1/chat/completions if present in the URL
            if base_url.endswith('/v1/chat/completions'):
                base_url = base_url[:-20]  # Remove the last 20 characters
            # Ensure base_url ends with a single slash
            base_url = base_url.rstrip('/') + '/'
            
            logger.info(f"Initializing local model with base_url: {base_url}")
            # Configure for local model
            self.llm = ChatOpenAI(
                model='local-model',  # Simple model name
                base_url=base_url + 'v1',  # Just add v1 to the base URL
                temperature=temp,
                api_key="not-needed",  # Required by LangChain but not used for local models
                default_headers={
                    "Content-Type": "application/json"
                }
            )
        else:
            # Use OpenAI model
            self.llm = ChatOpenAI(model='gpt-4o', temperature=temp)

    async def generate_answer(self, query: str, context: List[str]) -> str:
        logger.info(f"Generating answer for query: {query}")
        logger.info(f"Using context with {len(context)} segments")
        
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
            logger.info("Sending request to LLM...")
            answer = await chain.arun(query=query, context='\n\n'.join(context))
            logger.info("Received response from LLM")
            logger.info(f"Answer length: {len(answer)} characters")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            return f"An error occurred while generating the answer: {str(e)}"