---
name: nano-banana-cast
description: Edits a confirmed base video frame to insert or replace a character, product, or subject the user describes, while preserving the original background, lighting, and composition. Invoke only after a frame has already been confirmed via video search — never on a bare text request with no base image.
metadata:
  require-secret: true
  require-secret-description: Provide your Gemini API key — used to call the Nano Banana (Gemini image) model for the edit.
  homepage: internal
---

# Nano Banana Cast Skill (NB2 Lite)

## Purpose

Take a base frame the user already confirmed (from `search_video_library`)
and edit it to feature a character or product the user wants cast into that
scene — without regenerating the whole image from scratch. The background,
angle, and lighting of the original frame must be preserved; only the
described subject changes.

## When to invoke

- Only after `confirmed_frame` exists in session state.
- Never invoke speculatively to "try an idea" without an explicit user
  casting description.
- If the user asks to generate an image with no prior search/confirmation,
  ask them to find or pick a base frame first rather than inventing one.

## Required inputs

| field | type | required | notes |
|---|---|---|---|
| `base_frame_path` | image path | yes | the confirmed frame to edit |
| `casting_prompt` | string | yes | what to insert/replace, in the user's own words |
| `reference_image_path` | image path | no | a photo of the specific character/product, for visual identity consistency across multiple frames |

## Instructions for the orchestrator model

1. Confirm `base_frame_path` is set before calling this skill. If not, ask
   the user which moment to use first.
2. Compose the edit request sent to the image model as:
   ```
   Using the attached image as the base, keep the original background,
   camera angle, framing, and lighting unchanged. Add or replace the
   following into the scene: {casting_prompt}. Match the photorealistic
   quality and color grading of the base image so the edit is seamless.
   ```
3. If `reference_image_path` is present, append:
   ```
   Match the appearance of the subject shown in the second attached
   reference image as closely as possible.
   ```
4. Send `base_frame_path` (and `reference_image_path` if present) plus the
   composed prompt to the Gemini image endpoint (Nano Banana / NB2 Lite).
5. Return the edited frame to the user for approval before any video
   synthesis step runs. Do not chain directly into `synthesize_video`
   without an explicit confirmation.
6. If the user asks for a change, re-invoke with the same
   `base_frame_path` and a revised `casting_prompt` — do not edit the
   already-edited output, always branch from the original base frame to
   avoid compounding artifacts.

## Output contract

```
{ frame_path: string, frame_id: string }
```

## Failure handling

- API/model error: report it plainly in chat, keep the original unedited
  base frame available, offer one retry.
- Model refusal (e.g. real-person likeness, policy-restricted content):
  relay the refusal reason in plain language and ask the user to rephrase
  `casting_prompt` — do not silently substitute a different subject.
- Low-fidelity edit (background clearly altered from source): flag it to
  the user as a lower-confidence result rather than presenting it as final.
