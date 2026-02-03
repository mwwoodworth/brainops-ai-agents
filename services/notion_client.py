import os
import httpx
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class NotionClient:
    """
    Client for interacting with the Notion API to sync knowledge base articles.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }

    async def search_pages(self, query: str = "", page_size: int = 50) -> List[Dict[str, Any]]:
        """
        Search for pages in the workspace.
        """
        if not self.api_key:
            logger.warning("Notion API key not configured")
            return []

        async with httpx.AsyncClient() as client:
            try:
                payload = {
                    "filter": {"value": "page", "property": "object"},
                    "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                    "page_size": max(1, min(int(page_size or 50), 100))
                }
                if query:
                    payload["query"] = query

                response = await client.post(
                    f"{self.base_url}/search",
                    headers=self.headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Notion search failed: {response.text}")
                    return []

                data = response.json()
                return data.get("results", [])
            except Exception as e:
                logger.error(f"Notion search error: {e}")
                return []

    async def get_page_blocks(self, block_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve blocks (content) of a page.
        """
        if not self.api_key:
            return []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/blocks/{block_id}/children",
                    headers=self.headers,
                    timeout=30.0
                )
                if response.status_code != 200:
                    return []
                return response.json().get("results", [])
            except Exception as e:
                logger.error(f"Error fetching Notion blocks: {e}")
                return []

    def extract_title(self, page: Dict[str, Any]) -> str:
        """Helper to extract title from a Notion page object."""
        properties = page.get("properties", {})
        
        # Title is usually under 'Name', 'Title', or 'Page'
        for key, prop in properties.items():
            if prop.get("type") == "title":
                title_obj = prop.get("title", [])
                if title_obj:
                    return "".join([t.get("plain_text", "") for t in title_obj])
        
        return "Untitled Page"

    async def get_all_knowledge_docs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetch all pages and format them for the knowledge base.
        """
        safe_limit = max(1, min(int(limit or 50), 100))
        pages = await self.search_pages(page_size=safe_limit)
        docs = []
        
        for page in pages:
            try:
                doc = {
                    "id": page.get("id"),
                    "title": self.extract_title(page),
                    "url": page.get("url"),
                    "last_edited_time": page.get("last_edited_time"),
                    "created_time": page.get("created_time"),
                    "source": "notion"
                }
                docs.append(doc)
            except Exception as e:
                logger.warning(f"Failed to process Notion page {page.get('id')}: {e}")
                
        return docs
