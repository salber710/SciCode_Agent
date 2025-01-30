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

from scicode import keys_cfg_path
import config


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
        Generate 2-3 search queries to gather relevant background information.
        
        Guidelines for queries:
        - Use plain text without quotes or special characters
        - Keep each query under 10 words
        - Focus on simple tutorials and the fundamentals
        - Include both theoretical and practical aspects
        - Make queries specific but not too complex
        - Coding queries should be specific to Python
        
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

    def chunk_text(self, text: str, max_tokens: int = 4000) -> List[str]:
        """
        Split text into chunks that fit within token limits.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.encoder.encode(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            if current_length + 1 > max_tokens:
                chunks.append(self.encoder.decode(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(token)
            current_length += 1
        
        if current_chunk:
            chunks.append(self.encoder.decode(current_chunk))
        
        return chunks

    async def analyze_content(self, content: str, question: str) -> str:
        """
        Use GPT to analyze and summarize content relevant to the question.
        
        Args:
            content: Content to analyze
            question: Original scientific programming question
            
        Returns:
            Summarized analysis of the content
        """
        chunks = self.chunk_text(content)
        summaries = []
        
        for chunk in chunks:
            prompt = f"""
            Analyze this content in relation to the scientific programming question:
            Question: {question}
            
            Content: {chunk}
            
            Provide a concise summary of the relevant information, focusing on:
            1. Key concepts and theoretical background
            2. Implementation approaches and best practices
            3. Potential challenges and solutions
            """
            
            async with openai.AsyncOpenAI(api_key = self.openai_api_key) as client:
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a scientific research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            summaries.append(response.choices[0].message.content.strip())
        
        return "\n\n".join(summaries)

    async def generate_final_summary(self, 
                                   question: str, 
                                   sources: List[str]) -> Dict[str, str]:
        """
        Generate final summary and implementation guidance.
        
        Args:
            question: Original scientific programming question
            sources: List of contents relevant to the programming question
            
        Returns:
            Dictionary containing domain summary and implementation guidance
        """
        combined_sources = "\n\n".join([s for s in sources if len(s) < 10000])
        print(len(combined_sources))
        max_tokens_allowed = 7000


        ############ Summarize Chunks ##################################
        chunks = self.chunk_text(combined_sources, max_tokens=max_tokens_allowed)
        num_tokens_per_summary = max_tokens_allowed // len(chunks)
        summaries = []
        
        for chunk in chunks:
            prompt = f"""
            You will receive two pieces of information. 'Question:' which has the original question
            that we are tasked with and 'Sources:' which contains a bunch of information potentially
            relevant to answering the original question. Your job is to return a detailed and technical
            summary of the Sources. As best as possible, only remove content from Sources, do not paraphrase
            unless it is necessary. Remove information you find redundant or irrelevant to the task. In cases with conflicting information,
            include both sets of information, but note that they are conflicting. Ensure that your answer contains
            information regarding guidances on how to implement the code. Limit your summary to at most {num_tokens_per_summary}
            tokens. 
            
            Question: {question}
            
            Sources: {chunk}
            """
        
            async with openai.AsyncOpenAI(api_key = self.openai_api_key) as client:
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a scientific research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
            
            summary = response.choices[0].message.content.strip()
            summaries.append(summary)

        ##################### Blend Chunks Summaries into One Big Summary ###############################
        final_summaries = "\n\n".join(summaries)
        final_prompt = f"""
        You will receive two pieces of information. 'Question:' which has the original question
        that we are tasked with and 'Sources:' which contains a bunch of information potentially
        relevant to answering the original question. Your job is to return a detailed and technical
        summary of the Sources. As best as possible, only remove content from Sources, do not paraphrase
        unless it is necessary. Remove information you find redundant and in cases with conflicting information,
        include all of the sets of information, but note they are conflicting. Ensure that your answer contains
        information regarding guidances on how to implement the code. Limit your response to {max_tokens_allowed//4}  tokens.
        
        Question: {question}
        
        Sources: {final_summaries}
        """
        async with openai.AsyncOpenAI(api_key = self.openai_api_key) as client:
                final_response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a scientific research assistant."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.3
                )
            
        final_summary = final_response.choices[0].message.content.strip()

        
        return final_summary

    async def research_question(self, question: str) -> ResearchResult:
        """
        Main method to research and analyze a scientific programming question.
        
        Args:
            question: The scientific programming question to research
            
        Returns:
            ResearchResult containing the analysis and recommendations
        """
        # Generate search queries
        search_queries = await self.generate_search_queries(question)
        self.logger.info(f"Generated {len(search_queries)} search queries")
        
        # Perform web search
        search_results = await self.search_web(search_queries)
        self.logger.info(f"Found {len(search_results)} search results")
        
        # Fetch and analyze content
        analyses = []
        contents = []
        print(f"Iterating through {len(search_results)} search results...")
        for result in search_results:
            content = await self.fetch_and_parse_content(result['href'])
            #content = result['body']
            print(f"First 50/{len(content)} characters of content: {content[:50]}")
            if content:
                contents.append(content)
                #analysis = await self.analyze_content(content, question)
                #analyses.append(analysis)
        
        # Generate final summary
        summary = await self.generate_final_summary(question, contents)
        
        return ResearchResult(
            query=question,
            search_queries=search_queries,
            relevant_sources=search_results,
            summary=summary
        )
    
def get_config():
    if not keys_cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {keys_cfg_path}")
    return config.Config(str(keys_cfg_path))

# Example usage
async def main():
    # Initialize with your OpenAI API key
    key: str = get_config()["OPENAI_KEY"]
    assistant = ScientificResearchAssistant(key)
    
    # Example question
    question = "How do I implement adaptive mesh refinement for solving PDEs?"
    
    # Get research results
    result = await assistant.research_question(question)
    
    # Print results
    print(f"Research Results for: {result.query}\n")
    print("Search Queries Used:")
    for query in result.search_queries:
        print(f"- {query}")
    print("\nSummary:")
    print(result.summary)


if __name__ == "__main__":
    asyncio.run(main())