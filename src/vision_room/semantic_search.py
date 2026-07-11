import json
from dataclasses import dataclass
from typing import Any

from .embedding import SentenceTransformerEmbeddingModel
from .model_manager import ModelManager
from .search_tool import VideoSearchTool, SearchHit

@dataclass
class SemanticSearcher:
    model_manager: ModelManager
    embedding_model: SentenceTransformerEmbeddingModel
    search_tool: VideoSearchTool
    
    def _parse_query_with_gemma(self, query: str) -> dict[str, Any]:
        """Uses Local Gemma to parse natural language query into JSON schema."""
        if not self.model_manager.base_url:
            raise RuntimeError("ModelManager requires a base_url for semantic query parsing.")
            
        system_prompt = """You are a video search query parser. Given a user's natural language request to find a specific moment in a video, extract the search criteria into a structured JSON object.

Output strict JSON matching this schema and nothing else. If a field is not mentioned or implied in the user's query, leave it empty (empty string for strings, empty array for arrays, 0 for counts).

{
  "objects": ["distinct physical objects/items mentioned"],
  "activity": "the action or event they are looking for",
  "setting": "location/environment type mentioned",
  "colors": ["specific colors mentioned"],
  "people": {
    "count": 0,
    "description": "brief description of the people they are looking for"
  },
  "text_visible": "any specific text or signs they are searching for",
  "notable_details": ["any other specific disambiguating details"]
}

Output valid JSON only. Do not wrap in markdown fences."""

        import httpx
        payload = {
            "model": "gemma-4-local",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1
        }
        
        headers = {"Authorization": f"Bearer {self.model_manager.api_key}"} if self.model_manager.api_key else {}
        try:
            response = httpx.post(
                f"{self.model_manager.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            
            # Remove potential markdown fences
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
                
            return json.loads(content)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Query parsing failed: {e}. Falling back to raw query.")
            return {"activity": query} # Fallback to placing entire query in activity

    def search(self, raw_query: str, top_k: int = 3) -> list[SearchHit]:
        """Parses natural language query, generates search string, and executes vector search."""
        parsed_schema = self._parse_query_with_gemma(raw_query)
        
        # Concatenate exactly as done during ingestion
        activity = parsed_schema.get("activity", "")
        setting = parsed_schema.get("setting", "")
        objects = " ".join(parsed_schema.get("objects", []))
        people_desc = parsed_schema.get("people", {}).get("description", "")
        
        # activity + setting + " ".join(objects) + people.description
        search_blob = f"{activity} {setting} {objects} {people_desc}".strip()
        if not search_blob:
             # Failsafe if the parser returned entirely empty
             search_blob = raw_query
             
        return self.search_tool.search_video_library(search_blob, top_k=top_k)
