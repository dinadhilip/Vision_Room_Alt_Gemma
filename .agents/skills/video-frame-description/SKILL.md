---
name: video-frame-description
description: Produces a structured, embedding-ready description of a single video keyframe — objects, primary activity, setting, dominant colors, people, visible text, and disambiguating detail. Used at ingestion time to caption every extracted keyframe before embedding.
metadata:
  homepage: internal
---

# Video Frame Description Skill

## Purpose

Convert a single keyframe image into a consistent, structured caption that
EmbeddingGemma can embed and that a later text query can match against.
Consistency across frames matters more than creativity here — every frame
must be described using the same schema so embeddings live in a comparable
semantic space.

## When to invoke

Called once per extracted keyframe during ingestion, never during a live
user query. Not a conversational skill — no back-and-forth, single image in,
single structured object out.

## Model call

Run against Gemma 4 (E2B or E4B) vision input via `litert-lm serve` or direct
Python bindings. One image per call.

## Instruction body (system/user prompt sent with the image)

```
You are a visual indexing assistant. Given a single video keyframe, produce
a structured description used for semantic search indexing. Be literal and
specific. Describe only what is visibly present — do not infer narrative,
backstory, or emotional intent beyond what the image shows.

Output strict JSON matching this schema and nothing else — no preamble, no
markdown fences, no trailing commentary:

{
  "summary": string,        // one sentence, natural language, under 25 words,
                             // describing the overall scene as a human would
  "objects": string[],      // distinct physical objects/items visible,
                             // most prominent first, max 10
  "activity": string,       // the primary action or event taking place;
                             // use "static, no action" if nothing is happening
  "setting": string,        // location/environment type, e.g.
                             // "outdoor construction site", "indoor kitchen"
  "colors": string[],       // dominant colors present, max 5, common names
                             // only (e.g. "orange", not "burnt sienna")
  "people": {
    "count": integer,       // number of distinct humans visible, 0 if none
    "description": string   // brief, role/appearance only — never a name or
                             // identity claim, e.g. "two workers in hi-vis
                             // vests"; empty string if count is 0
  },
  "text_visible": string,   // any readable text/signage/labels in frame,
                             // verbatim if legible; empty string if none
  "notable_details": string[] // 1-3 specific details that would help
                             // disambiguate this frame from a visually
                             // similar one from the same video
}

Rules:
- Never guess the identity of a real person. Describe appearance and role
  only.
- If you are not confident about an attribute, omit it rather than guess.
- Keep total output under 150 words.
- Output valid JSON only.
```

## Output contract

```
{ frame_id: string, caption_json: <schema above> }
```

## Downstream use (embed.py)

Concatenate `summary + activity + setting + " ".join(objects) +
people.description` into one text blob per frame before embedding — this is
the string EmbeddingGemma indexes. Keep `colors`, `text_visible`, and
`notable_details` stored alongside the vector as filterable metadata rather
than embedded, since they're better used for re-ranking or exact-match
filtering than semantic similarity.

## Failure handling

If the model returns non-JSON or a partial object, retry once with the same
image and prompt. On a second failure, store the frame with `summary` set to
the model's raw text output and all other fields empty, so it's still
searchable by keyword rather than dropped from the index entirely.
