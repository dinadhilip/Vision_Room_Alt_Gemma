---
name: video-query-parser
description: Converts a natural language video search query into a structured JSON object matching the video frame schema. Used to prepare user queries for semantic vector search and metadata filtering against the indexed keyframes.
metadata:
  homepage: internal
---

# Video Query Parser Skill

## Purpose

Convert a natural language search query (e.g. "find the part where the blue car drives past a building") into a structured JSON object. This structure must perfectly align with the schema used by `video-frame-description` during ingestion, allowing us to generate an equivalent embedding string to compare against our vector index.

## When to invoke

Called whenever the user asks to search for a specific moment, object, person, or event in the ingested video library.

## Required inputs

| field | type | required | notes |
|---|---|---|---|
| `query` | string | yes | The natural language query from the user. |

## Instruction body (system/user prompt)

```
You are a video search query parser. Given a user's natural language request to find a specific moment in a video, extract the search criteria into a structured JSON object.

Output strict JSON matching this schema and nothing else. If a field is not mentioned or implied in the user's query, leave it empty (empty string for strings, empty array for arrays, 0 for counts).

{
  "objects": string[],      // distinct physical objects/items mentioned
  "activity": string,       // the action or event they are looking for
  "setting": string,        // location/environment type mentioned
  "colors": string[],       // specific colors mentioned
  "people": {
    "count": integer,       // number of distinct humans mentioned, 0 if none
    "description": string   // brief description of the people they are looking for
  },
  "text_visible": string,   // any specific text or signs they are searching for
  "notable_details": string[] // any other specific disambiguating details
}

Example Query: "Find the part where two guys in yellow hardhats are pouring concrete"
Example Output:
{
  "objects": ["concrete", "hardhats"],
  "activity": "pouring concrete",
  "setting": "",
  "colors": ["yellow"],
  "people": {
    "count": 2,
    "description": "two guys in hardhats"
  },
  "text_visible": "",
  "notable_details": []
}

Output valid JSON only. Do not wrap in markdown fences.
```

## Downstream use (search.py)

Once parsed, concatenate the fields `activity + setting + " ".join(objects) + people.description` into one text blob exactly as done during ingestion.
Embed this concatenated text using `SentenceTransformerEmbeddingModel` to generate the query vector.
Execute a cosine similarity search against the SQLite/Vector index. Use `colors`, `text_visible`, and `notable_details` for exact-match or re-ranking filtering if necessary.
