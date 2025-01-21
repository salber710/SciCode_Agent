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
from scicode import keys_cfg_path

@dataclass
class ResearchResult:
    """Contains the research results for a scientific programming question."""
    query: str
    search_queries: List[str]
    relevant_sources: List[Dict[str, str]]
    domain_summary: str
    implementation_guidance: str

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
        Generate 3-5 specific search queries to gather relevant background information.
        Focus on academic sources, documentation, and technical references.
        Return only the queries, one per line.
        """
        
        async with openai.AsyncOpenAI() as client:
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

    async def search_web(self, queries: List[str], max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web for relevant information using DuckDuckGo.
        
        Args:
            queries: List of search queries
            max_results: Maximum number of results per query
            
        Returns:
            List of dictionaries containing search results
        """
        results = []
        ddgs = DDGS()
        
        for query in queries:
            try:
                # DuckDuckGo search is synchronous, we'll run it in the executor
                search_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: list(ddgs.text(query, max_results=max_results))
                )
                results.extend(search_results)
            except Exception as e:
                self.logger.error(f"Error searching for {query}: {str(e)}")
        
        return results

    async def fetch_and_parse_content(self, url: str) -> str:
        """
        Fetch and parse content from a URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Parsed text content from the webpage
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        
                        # Clean up whitespace
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {str(e)}")
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
            
            async with openai.AsyncOpenAI() as client:
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
                                   analyses: List[str]) -> Dict[str, str]:
        """
        Generate final summary and implementation guidance.
        
        Args:
            question: Original scientific programming question
            analyses: List of content analyses
            
        Returns:
            Dictionary containing domain summary and implementation guidance
        """
        combined_analyses = "\n\n".join(analyses)
        
        prompt = f"""
        Based on the following analyses of a scientific programming question,
        provide two separate sections:
        
        Question: {question}
        
        Analyses:
        {combined_analyses}
        
        1. Domain Summary: Theoretical background and key concepts
        2. Implementation Guidance: Practical steps, best practices, and code considerations
        
        Be specific and technical, but concise.
        """
        
        async with openai.AsyncOpenAI() as client:
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a scientific research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
        
        summary = response.choices[0].message.content.strip()
        
        # Split into sections
        sections = summary.split('\n\n')
        domain_summary = sections[0].replace("Domain Summary:", "").strip()
        implementation_guidance = sections[1].replace("Implementation Guidance:", "").strip()
        
        return {
            "domain_summary": domain_summary,
            "implementation_guidance": implementation_guidance
        }

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
        for result in search_results:
            content = await self.fetch_and_parse_content(result['link'])
            if content:
                analysis = await self.analyze_content(content, question)
                analyses.append(analysis)
        
        # Generate final summary
        summary = await self.generate_final_summary(question, analyses)
        
        return ResearchResult(
            query=question,
            search_queries=search_queries,
            relevant_sources=search_results,
            domain_summary=summary['domain_summary'],
            implementation_guidance=summary['implementation_guidance']
        )

# Example usage
async def main():
    # Initialize with your OpenAI API key
    assistant = ScientificResearchAssistant(os.getenv("OPENAI_API_KEY"))
    
    # Example question
    question = "How do I implement adaptive mesh refinement for solving PDEs?"
    
    # Get research results
    result = await assistant.research_question(question)
    
    # Print results
    print(f"Research Results for: {result.query}\n")
    print("Search Queries Used:")
    for query in result.search_queries:
        print(f"- {query}")
    print("\nDomain Summary:")
    print(result.domain_summary)
    print("\nImplementation Guidance:")
    print(result.implementation_guidance)

if __name__ == "__main__":
    asyncio.run(main())