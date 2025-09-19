import requests
import time
import logging
from typing import List, Dict, Any, Optional
import asyncio
from firecrawl import AsyncFirecrawlApp
from mistralai import Mistral
import json
import re
import os
from fastmcp import FastMCP
# Initialize FastMCP server
mcp = FastMCP("semantic-scholar-mcp")

api_key = ""
client = Mistral(api_key=api_key)
# Base URL for the Semantic Scholar API
BASE_URL = "https://api.semanticscholar.org/graph/v1"
BASE_RECOMMENDATION_URL = "https://api.semanticscholar.org/recommendations/v1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def bibtex_helper(html_content):
    # Extract bibtex citation from HTML content
    bibtex_pattern = r'<pre class="bibtex-citation"[^>]*>(.*?)</pre>'
    bibtex_match = re.search(bibtex_pattern, html_content, re.DOTALL)

    if bibtex_match:
        bibtex_citation = bibtex_match.group(1).strip()
        logger.info("Extracted BibTeX citation: %s", bibtex_citation)
        return bibtex_citation
    else:
        logger.warning("BibTeX citation not found in HTML content")
        return None
def get_paper_citations(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get citations for a specific paper."""
    url = f"{BASE_URL}/paper/{paper_id}/citations"
    params = {
        "limit": min(limit, 100),  # API limit is 100
        "fields": "contexts,isInfluential,title,authors,year,venue"
    }
    
    try:
        response_data = make_request_with_retry(url, params=params)
        citations = response_data.get("data", [])
        
        return [
            {
                "contexts": citation.get("contexts", []),
                "isInfluential": citation.get("isInfluential"),
                "citingPaper": {
                    "paperId": citation.get("citingPaper", {}).get("paperId"),
                    "title": citation.get("citingPaper", {}).get("title"),
                    "authors": [{"name": author.get("name"), "authorId": author.get("authorId")} 
                               for author in citation.get("citingPaper", {}).get("authors", [])],
                    "year": citation.get("citingPaper", {}).get("year"),
                    "venue": citation.get("citingPaper", {}).get("venue")
                }
            } for citation in citations
        ]
    except Exception as e:
        logger.error(f"Error getting citations for {paper_id}: {e}")
        return []

def get_paper_references(paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get references for a specific paper."""
    url = f"{BASE_URL}/paper/{paper_id}/references"
    params = {
        "limit": min(limit, 100),  # API limit is 100
        "fields": "contexts,isInfluential,title,authors,year,venue"
    }
    
    try:
        response_data = make_request_with_retry(url, params=params)
        references = response_data.get("data", [])
        
        return [
            {
                "contexts": reference.get("contexts", []),
                "isInfluential": reference.get("isInfluential"),
                "citedPaper": {
                    "paperId": reference.get("citedPaper", {}).get("paperId"),
                    "title": reference.get("citedPaper", {}).get("title"),
                    "authors": [{"name": author.get("name"), "authorId": author.get("authorId")} 
                               for author in reference.get("citedPaper", {}).get("authors", [])],
                    "year": reference.get("citedPaper", {}).get("year"),
                    "venue": reference.get("citedPaper", {}).get("venue")
                }
            } for reference in references
        ]
    except Exception as e:
        logger.error(f"Error getting references for {paper_id}: {e}")
        return []

def get_citations_and_references(paper_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get citations and references for a paper using paper ID."""
    citations = get_paper_citations(paper_id)
    references = get_paper_references(paper_id)
    
    return {
        "citations": citations,
        "references": references
    }


def make_request_with_retry(url: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None, 
                           method: str = "GET", max_retries: int = 5, base_delay: float = 1.0) -> Dict[str, Any]:
    """
    Make HTTP request with retry logic for 429 rate limit errors.
    
    Args:
        url: The URL to make the request to
        params: Query parameters for GET requests
        json_data: JSON data for POST requests
        method: HTTP method (GET or POST)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds, will be exponentially increased
    
    Returns:
        JSON response as dictionary
    
    Raises:
        Exception: If all retries are exhausted or other errors occur
    """
    
    for attempt in range(max_retries + 1):
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json=json_data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check if request was successful
            if response.status_code == 200:
                return response.json()
            
            # Handle rate limiting (429 Too Many Requests)
            elif response.status_code == 429:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit (429). Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded. Max retries ({max_retries}) exhausted.")
            
            # Handle other HTTP errors
            else:
                response.raise_for_status()
                
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request timeout. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            else:
                raise Exception("Request timeout. Max retries exhausted.")
        
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request failed: {e}. Retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(delay)
                continue
            else:
                raise Exception(f"Request failed after {max_retries} retries: {e}")
    
    raise Exception("Unexpected error in request retry logic")

def search_papers(query: str, limit: int = 1) -> List[Dict[str, Any]]:
    """Search for papers using a query string."""
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),  # API limit is 100
        "fields": "paperId,title,abstract,url,venue,publicationTypes,citations,citationCount,tldr"
    }
    
    try:
        response_data = make_request_with_retry(url, params=params)
        papers = response_data.get("data", [])
        return [
            {
                "paperId": paper.get("paperId"),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "publicationTypes": paper.get("publicationTypes"),

            } for paper in papers
        ]
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        return []


@mcp.tool()
async def search_semantic_scholar(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search for papers on Semantic Scholar using a query string.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)

    Returns:
        List of dictionaries containing paper information
    """
    logging.info(f"Searching for papers with query: {query}, num_results: {num_results}")
    try:
        results = await asyncio.to_thread(search_papers, query, num_results)
        return results
    except Exception as e:
        return [{"error": f"An error occurred while searching: {str(e)}"}]

async def get_semantic_scholar_citations_and_references(paper_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get citations and references for a specific paper on Semantic Scholar.

    Args:
        paper_id: ID of the paper

    Returns:
        Dictionary containing lists of citations and references
    """
    logging.info(f"Fetching citations and references for paper ID: {paper_id}")
    try:
        citations_refs = await asyncio.to_thread(get_citations_and_references, paper_id)
        return citations_refs
    except Exception as e:
        return {"error": f"An error occurred while fetching citations and references: {str(e)}"}

@mcp.tool()
async def read_paper(paper_id: str):
    """
    Perform comprehensive deep analysis of a research paper by extracting full content and metadata.
    
    **IMPORTANT: Use this function ONLY when you need detailed paper content analysis.**
    This is a resource-intensive operation that performs OCR on the entire PDF document.
    
    Use cases for this function:
    - When you need to analyze specific sections, methodologies, or results within the paper
    - When you need to extract detailed technical content, equations, or experimental data
    - When you need to understand the paper's full context beyond just the abstract
    - When preparing detailed summaries or conducting thorough literature reviews
    
    Do NOT use this function for:
    - Simple paper discovery or browsing (use search_semantic_scholar instead)
    - Getting basic paper information like title, abstract, or citation count
    - Quick paper screening or filtering
    
    Args:
        paper_id: The Semantic Scholar paper ID (e.g., from search results)
    
    Returns:
        Tuple containing:
        - List of dictionaries with OCR-processed page content from the PDF
        - BibTeX citation string for the paper
    """
    url = f"https://www.semanticscholar.org/paper/{paper_id}"
    
    try:
        app = AsyncFirecrawlApp(api_key='')
        response = await app.scrape_url(
            url=url,		
            formats= [ 'html' ],
            # only_main_content= True,
            # parse_pdf= True,
            max_age= 14400000
        )
        data = json.loads(response.model_dump_json())
        # get the url 
        pdf_url = data['metadata']['citation_pdf_url']
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url
            },
            include_image_base64=False,
        )
        return json.loads(ocr_response.model_dump_json())["pages"] , bibtex_helper(data['html'])
    except Exception as e:
        return {"error": f"An error occurred while read paper: {str(e)}"} ,None


# RUNNING THE SERVER
def main():
    """Main entry point for the MCP server."""
    # logging.info("Starting Semantic Scholar MCP server")
    mcp.run()

if __name__ == "__main__":
    main()
