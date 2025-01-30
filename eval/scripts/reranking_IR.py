import os
from typing import List, Dict, Any
import aiohttp
import asyncio
from dataclasses import dataclass
import openai
from duckduckgo_search import DDGS
import logging
from bs4 import BeautifulSoup
import tiktoken
import re
import json
import time
import pickle
import aiohttp
from datetime import datetime

#from langchain.docstore.document import Document
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import CrossEncoder

from helper_functions import *
from evaluate_rag import *

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
TEMP = {}

class CrossEncoderRetriever(BaseRetriever, BaseModel):
    vectorstore: Any = Field(description="Vector store for initial retrieval")
    cross_encoder: Any = Field(description="Cross-encoder model for reranking")
    k: int = Field(default=5, description="Number of documents to retrieve initially")
    rerank_top_k: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str):
        # Initial retrieval
        initial_docs = self.vectorstore.similarity_search(query, k=self.k)
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort documents by score
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        # Return top reranked documents
        return [doc for doc, _ in scored_docs[:self.rerank_top_k]]

    async def aget_relevant_documents(self, query: str):
        raise NotImplementedError("Async retrieval not implemented")

@dataclass
class ResearchResult:
    """Contains the research results for a scientific programming question."""
    query: str
    search_queries: List[str]
    relevant_sources: List[Dict[str, str]]
    summary: str

class ScientificResearchAssistant:
    """
    Assists with scientific programming questions by combining web search
    and GPT-based analysis.
    """
    
    def __init__(self, openai_api_key: str):
        """
        Initialize the research assistant with necessary API keys.
        
        Args:
            openai_api_key: OpenAI API key for GPT model access
        """
        self.openai_api_key = openai_api_key
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def generate_search_queries(self, question: str) -> List[str]:
        """
        Use GPT to generate relevant search queries based on the question.
        
        Args:
            question: The scientific programming question
            
        Returns:
            List of search queries to gather information
        """
        prompt = f"""
        Given this scientific programming question: "{question}"
        Generate 3-4 search queries to gather relevant background information.
        
        Guidelines for queries:
        - Use plain text without quotes or special characters
        - Keep each query under 10 words
        - Focus on getting background information, not on coding
        - Make queries specific but not too complex]
        - Focus on the current step
        
        Return only the queries, one per line.
        """
        
        async with openai.AsyncOpenAI(api_key = self.openai_api_key) as client:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
        
        queries = response.choices[0].message.content.strip().split('\n')
        return [q.strip() for q in queries if q.strip()]

    async def clean_query(self, query: str) -> str:
        """
        Clean and format search query by:
        - Removing enumeration (1., 2., etc)
        - Removing quotes
        - Removing extra whitespace
        
        Args:
            query: Raw search query
            
        Returns:
            Cleaned query string
        """
        # Remove enumeration
        query = re.sub(r'^\d+\.\s*', '', query)
        # Remove quotes
        query = query.replace('"', '').replace('"', '').replace('"', '')
        # Clean whitespace
        query = ' '.join(query.split())
        return query

    async def search_web(self, queries: List[str], max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for relevant information using DuckDuckGo.
        
        Args:
            queries: List of search queries
            max_results: Maximum number of results per query
            
        Returns:
            List of dictionaries containing search results
        """
        results = []

        seen_hrefs = set() # links already included
        
        for raw_query in queries:
            try:
                # Clean the query
                query = await self.clean_query(raw_query)
                self.logger.info(f"Searching with cleaned query: {query}")
                
                # Create a new DDGS instance for each query
                ddgs = DDGS()
                
                # Run the synchronous search in the default thread pool
                loop = asyncio.get_running_loop()
                search_results = await loop.run_in_executor(
                    None,
                    lambda: list(ddgs.text(query, max_results=max_results))
                )
                
                self.logger.info(f"Found {len(search_results)} results for query: {query}")
                
                if search_results:
                    print(search_results)
                    results.extend(result for result in search_results 
                                    if result.get('href') 
                                    and not seen_hrefs.add(result['href']))
                else:
                    # If no results, try a more general version of the query
                    simplified_query = ' '.join(query.split()[:4])  # Take first 4 words
                    self.logger.info(f"Retrying with simplified query: {simplified_query}")
                    
                    search_results = await loop.run_in_executor(
                        None,
                        lambda: list(ddgs.text(simplified_query, max_results=max_results))
                    )
                    if search_results:
                        results.extend(result for result in search_results 
                                        if result.get('href') 
                                        and not seen_hrefs.add(result['href']))
                
                # Add a small delay between queries to avoid rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error searching for query '{raw_query}': {str(e)}")
        
        return results

    async def search_wiki(self, queries: List[str], max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search Wikipedia for relevant information.
        
        Args:
            queries: List of search queries
            max_results: Maximum number of results per query
            
        Returns:
            List of dictionaries containing search results
        """
        results = []
        seen_titles = set()  # titles already included
        
        async with aiohttp.ClientSession() as session:
            for raw_query in queries:
                try:
                    # Clean the query
                    query = await self.clean_query(raw_query)
                    self.logger.info(f"Searching Wikipedia with cleaned query: {query}")
                    
                    # Construct Wikipedia API URL
                    params = {
                        "action": "query",
                        "format": "json",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": max_results,
                        "srprop": "snippet|titlesnippet",
                    }
                    
                    # Make API request
                    async with session.get(
                        "https://en.wikipedia.org/w/api.php",
                        params=params
                    ) as response:
                        data = await response.json()
                        
                        search_results = data.get("query", {}).get("search", [])
                        self.logger.info(f"Found {len(search_results)} Wikipedia results for query: {query}")
                        
                        if search_results:
                            # Transform results into the expected format
                            for result in search_results:
                                if result["title"] not in seen_titles:
                                    seen_titles.add(result["title"])
                                    results.append({
                                        "title": result["title"],
                                        "body": result["snippet"],
                                        "href": f"https://en.wikipedia.org/wiki/{result['title'].replace(' ', '_')}",
                                    })
                        else:
                            # If no results, try a more general version of the query
                            simplified_query = ' '.join(query.split()[:4])  # Take first 4 words
                            self.logger.info(f"Retrying with simplified query: {simplified_query}")
                            
                            params["srsearch"] = simplified_query
                            async with session.get(
                                "https://en.wikipedia.org/w/api.php",
                                params=params
                            ) as retry_response:
                                retry_data = await retry_response.json()
                                retry_results = retry_data.get("query", {}).get("search", [])
                                
                                for result in retry_results:
                                    if result["title"] not in seen_titles:
                                        seen_titles.add(result["title"])
                                        results.append({
                                            "title": result["title"],
                                            "body": result["snippet"],
                                            "href": f"https://en.wikipedia.org/wiki/{result['title'].replace(' ', '_')}",
                                        })
                    
                    # Add a small delay between queries
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error searching Wikipedia for query '{raw_query}': {str(e)}")
        
        return results

    async def fetch_and_parse_content(self, url: str) -> str:
        """
        Fetch and parse content from a URL, handling both HTML and PDF files.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Parsed text content from the webpage or PDF
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        # Check if it's a PDF by examining content type
                        content_type = response.headers.get('Content-Type', '').lower()
                        
                        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                            # Handle PDF content
                            pdf_content = await response.read()
                            
                            # Run PDF parsing in a thread pool to avoid blocking
                            loop = asyncio.get_running_loop()
                            text = await loop.run_in_executor(
                                None,
                                self.parse_pdf_content,
                                pdf_content
                            )
                            return text
                        else:
                            # Handle HTML content
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Remove script and style elements
                            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                                element.decompose()
                            
                            # Get text content
                            text = soup.get_text()
                            
                            # Clean up whitespace
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            
                            return text
                    else:
                        self.logger.error(f"Failed to fetch {url}: Status {response.status}")
                        return ""
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
                return ""
            
    def parse_pdf_content(self, pdf_content: bytes) -> str:
        """
        Parse PDF content using PyPDF2.
        
        Args:
            pdf_content: Raw PDF content in bytes
            
        Returns:
            Extracted text from the PDF
        """
        try:
            import io
            from PyPDF2 import PdfReader
            
            # Create a PDF reader object
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_file)
            
            # Extract text from all pages
            text_content = []
            for page in pdf_reader.pages:
                text_content.append(page.extract_text())
            
            # Combine text and clean up whitespace
            text = ' '.join(text_content)
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            return ""
        
    async def encode_websites(self, website_contents: List[str]) -> FAISS:
        """
        Encodes multiple website contents into a vector store.
        
        Args:
            website_contents: List of strings or coroutines that resolve to strings
            
        Returns:
            A FAISS vector store containing the encoded website contents
        """
        # First, ensure all contents are strings
        processed_contents = []
        for content in website_contents:
            if asyncio.iscoroutine(content):
                content = await content
            processed_contents.append(content)

        # Create documents
        documents = [
            {
                'page_content': content,
                'metadata': {'source': f'website_{i}'}
            }
            for i, content in enumerate(processed_contents)
        ]

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = text_splitter.create_documents(
            [doc['page_content'] for doc in documents],
            metadatas=[doc['metadata'] for doc in documents]
        )

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)

        return vectorstore
    
    async def research_question(self, question: str) -> ResearchResult:
        """
        Main method to research and analyze a scientific programming question.
        
        Args:
            question: The scientific programming question to research
            
        Returns:
            ResearchResult containing the analysis and recommendations
        """
        # Generate search queries
        #print("Starting search queries...")
        print(question)
        search_queries = await self.generate_search_queries(question)
        self.logger.info(f"Generated {len(search_queries)} search queries")
        print(search_queries)
        
        # Perform web search
        search_results = await self.search_wiki(search_queries)
        self.logger.info(f"Found {len(search_results)} search results")
        print(search_results)

        '''
        website_contents = []
        for result in search_results:
            content = await self.fetch_and_parse_content(result['href'])
            website_contents.append(content)
        self.logger.info(f"Extracted {len(website_contents)} website contents")

        vectorstore = await self.encode_websites(website_contents)
        self.logger.info(f"Encoded website contents")


        ##################### RETRIEVE #############################
        cross_encoder_retriever = CrossEncoderRetriever(
            vectorstore=vectorstore,
            cross_encoder=self.cross_encoder,
            k=15,  # Retrieve 15 documents initially
            rerank_top_k=5  # Return top 3 after reranking
        )

        reranked_docs = cross_encoder_retriever.get_relevant_documents(question)
        context = '\n\n'.join(result.page_content for result in reranked_docs)
        #print(context)
        ###############################################################

        return context
        '''
        TEMP[question] = [search_queries, search_results]

    
async def process_jsonl(input_path, output_path, assistant):
    # Load progress if exists
    progress_path = output_path + '.progress'
    start_idx = 0
    processed_problems = []
    
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
            start_idx = progress['last_problem_idx']
            with open(output_path, 'r') as out_f:
                processed_problems = [json.loads(line) for line in out_f]
        print(f"Resuming from problem {start_idx}")
    
    # Read input file
    with open(input_path, 'r') as file:
        problems = [json.loads(line) for line in file]
    
    # Process each problem
    for problem_idx in range(start_idx, len(problems)):
        problem = problems[problem_idx]
        print(f"Processing problem {problem_idx}/{len(problems)}...")
        
        # Process each substep
        for substep_idx, subproblem in enumerate(problem['sub_steps']):
            start_substep = time.time()
            print(f"Processing substep {substep_idx}/{len(problem['sub_steps'])}...")
            query = f"The overall goal of our script is to: {problem['problem_description_main']} The current step is to: {subproblem['step_description_prompt']}"
            
            try:
                # Get research results
                subproblem['step_background'] = await assistant.research_question(query)
                
                # Save timestamp of when this was processed
                subproblem['processed_timestamp'] = datetime.now().isoformat()
                
            except Exception as e:
                print(f"Error processing problem {problem_idx}, substep {substep_idx}: {str(e)}")
                # Save error information
                subproblem['processing_error'] = str(e)
            print(f"Finished in {time.time() - start_substep} seconds...")
        
        # Add processed problem to list
        processed_problems.append(problem)

        with open("temp.pickle", "wb") as f:
            pickle.dump(TEMP, f)
        
        # Save progress every problem
        with open(progress_path, 'w') as f:
            json.dump({
                'last_problem_idx': problem_idx,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        # Save current results
        with open(output_path, 'w') as f:
            for p in processed_problems:
                json.dump(p, f)
                f.write('\n')
        
        print(f"Saved progress after problem {problem_idx}")

async def main():
    # Initialize with your OpenAI API key
    key: str = os.environ["OPENAI_API_KEY"]
    assistant = ScientificResearchAssistant(key)

    try:
        await process_jsonl('eval/data/problems_all.jsonl', 
                            'eval/data/problems_all_RAG_wiki.jsonl',
                            assistant)
    except Exception as e:
        print(f"Process interrupted: {str(e)}")
        print("Progress has been saved and can be resumed by running the script again")


if __name__ == "__main__":
    asyncio.run(main())