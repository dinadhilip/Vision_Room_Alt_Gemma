from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from .providers import CastProvider, VideoProvider
from .local_agent import LocalGemmaPlanner, PlannedReply, PlannedToolCall
from .semantic_search import SemanticSearcher
from .search_tool import SearchHit, VideoSearchTool


SYSTEM_PROMPT = """You are a conversational video assistant. Use search_video_library for local footage search, cast_into_frame after a confirmed frame, and synthesize_video after an approved anchor frame. Keep replies short and never expose tool JSON to the user."""

APPROVAL_RE = re.compile(r"\b(approve|approved|looks good|use this|go ahead|make video|synthesize|render)\b", re.I)
REVISION_RE = re.compile(r"\b(change|revise|instead|make it|swap|faster|slower|edit)\b", re.I)
CAST_RE = re.compile(r"\b(cast|put|place|add|feature|with|wearing|holding|product|character)\b", re.I)
SELECTION_RE = re.compile(
    r"\b(?:use|select|choose|confirm|pick|swap(?:\s+in)?)(?:\s+the)?\s+"
    r"(?P<choice>first|second|third|fourth|fifth|last|\d+)(?:st|nd|rd|th)?\b",
    re.I,
)
ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class SessionState:
    session_id: str
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    conversation_history: list[ChatMessage] = field(default_factory=list)
    matched_frames: list[str] = field(default_factory=list)
    confirmed_frame: str | None = None
    casting_prompt: str | None = None
    anchor_frames: list[str] = field(default_factory=list)
    video_history: list[dict] = field(default_factory=list)
    last_hits: list[SearchHit] = field(default_factory=list)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str | None) -> SessionState:
        resolved_id = session_id or uuid.uuid4().hex
        if resolved_id not in self._sessions:
            self._sessions[resolved_id] = SessionState(session_id=resolved_id)
        return self._sessions[resolved_id]

    def get_existing(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> SessionState:
        self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def count(self) -> int:
        return len(self._sessions)


class AgentOrchestrator:
    """Rule-based agent loop with the same tool contract the Gemma loop will use."""

    def __init__(
        self,
        searcher: SemanticSearcher,
        cast_provider: CastProvider,
        video_provider: VideoProvider,
        *,
        search_confidence_threshold: float = 0.20,
        local_planner: LocalGemmaPlanner | None = None,
    ) -> None:
        self.searcher = searcher
        self.cast_provider = cast_provider
        self.video_provider = video_provider
        self.search_confidence_threshold = search_confidence_threshold
        self.local_planner = local_planner

    def handle_turn(self, session: SessionState, message: str) -> dict:
        with session.lock:
            return self._handle_turn_locked(session, message)

    def _handle_turn_locked(self, session: SessionState, message: str) -> dict:
        clean_message = message.strip()
        session.conversation_history.append(ChatMessage(role="user", content=clean_message))

        planned_response = self._try_local_planner(session)
        if planned_response is not None:
            response = planned_response
        elif selection := self._selection_from_message(session, clean_message):
            response = self.confirm_frame(session, selection)
        elif self._should_synthesize(session, clean_message):
            response = self._synthesize(session, clean_message)
        elif APPROVAL_RE.search(clean_message) and session.matched_frames and not session.anchor_frames:
            response = {
                "reply": "I need an approved anchor frame before video synthesis. Tell me who or what to cast into the matched frame.",
                "ui_action": {"type": "none", "payload": {}},
            }
        elif self._should_cast(session, clean_message):
            response = self._cast(session, clean_message)
        else:
            response = self._search(session, clean_message)

        session.conversation_history.append(ChatMessage(role="assistant", content=response["reply"]))
        response["session_id"] = session.session_id
        response["state"] = self._public_state(session)
        return response

    def _try_local_planner(self, session: SessionState) -> dict | None:
        if self.local_planner is None:
            return None
        try:
            decision = self.local_planner.plan(
                system_prompt=SYSTEM_PROMPT,
                history=[
                    {"role": message.role, "content": message.content}
                    for message in session.conversation_history[-12:]
                ],
                state=self._public_state(session),
            )
        except Exception:
            return None

        if isinstance(decision, PlannedReply):
            return {"reply": decision.reply, "ui_action": {"type": "none", "payload": {}}}
        if isinstance(decision, PlannedToolCall):
            return self._execute_planned_tool(session, decision)
        return None

    def _execute_planned_tool(self, session: SessionState, decision: PlannedToolCall) -> dict | None:
        args = decision.arguments
        if decision.name == "search_video_library":
            return self._search(session, str(args.get("query") or ""), int(args.get("top_k") or 3))
        if decision.name == "cast_into_frame":
            casting_prompt = str(args.get("casting_prompt") or "").strip()
            if not casting_prompt:
                return None
            base_frame_path = str(args.get("base_frame_path") or "").strip()
            if base_frame_path:
                result = self.cast_provider.cast_into_frame(
                    base_frame_path,
                    casting_prompt,
                    args.get("reference_image_path"),
                )
                session.casting_prompt = casting_prompt
                session.anchor_frames = [result["frame_path"]]
                return self._cast_response(result, casting_prompt)
            return self._cast(session, casting_prompt)
        if decision.name == "synthesize_video":
            anchor_paths = args.get("anchor_frame_paths")
            if isinstance(anchor_paths, list) and anchor_paths:
                session.anchor_frames = [str(path) for path in anchor_paths]
            narrative_hint = str(args.get("narrative_hint") or "").strip()
            if narrative_hint:
                return self._synthesize_with_args(
                    session,
                    narrative_hint=narrative_hint,
                    duration_hint_s=int(args.get("duration_hint_s") or 15),
                    edit_instruction=args.get("edit_instruction"),
                    prior_video_id=args.get("prior_video_id"),
                )
        if decision.name == "generate_storyboard":
            story = str(args.get("story") or "").strip()
            if story:
                return self._generate_storyboard(session, story, args.get("style", "comic"))
        return None

    def handle_confirm_frame(self, session: SessionState, frame_id: str) -> dict:
        with session.lock:
            response = self.confirm_frame(session, frame_id)
            response["session_id"] = session.session_id
            response["state"] = self._public_state(session)
            return response

    def confirm_frame(self, session: SessionState, frame_id: str) -> dict:
        selected = None
        for hit in session.last_hits:
            if hit.record.id == frame_id:
                selected = hit
                break

        if selected is None:
            return {
                "reply": "I could not find that frame in the current result set. Search again or pick one of the visible matches.",
                "ui_action": {"type": "none", "payload": {}},
            }

        session.confirmed_frame = selected.record.id
        session.casting_prompt = None
        session.anchor_frames = []
        session.video_history = []
        return {
            "reply": f"Selected the frame at {selected.record.timestamp_s:.1f}s. Tell me who or what to cast into it.",
            "ui_action": {
                "type": "show_frame_gallery",
                "payload": {
                    "primary": selected.to_public_dict(),
                    "frames": [hit.to_public_dict() for hit in session.last_hits],
                    "confirmed_frame": selected.record.id,
                },
            },
        }

    def _search(self, session: SessionState, query: str, top_k: int = 3) -> dict:
        hits = self.searcher.search(query, top_k=top_k)
        session.last_hits = hits
        session.matched_frames = [hit.record.id for hit in hits]
        session.confirmed_frame = hits[0].record.id if hits else None
        session.casting_prompt = None
        session.anchor_frames = []
        session.video_history = []

        if not hits:
            return {
                "reply": "I could not find indexed local footage yet. Add frames or run ingestion, then ask again.",
                "ui_action": {"type": "none", "payload": {}},
            }

        top = hits[0]
        if top.score < self.search_confidence_threshold:
            reply = (
                "I found a weak possible match, but I would not treat it as certain. "
                "You can still use it as a speculative starting point."
            )
        else:
            reply = (
                f"I found a likely moment around {top.record.timestamp_s:.1f}s: "
                f"{top.record.caption} Want to cast someone or a product into this frame?"
            )

        return {
            "reply": reply,
            "ui_action": {
                "type": "show_frame_gallery",
                "payload": {
                    "primary": top.to_public_dict(),
                    "frames": [hit.to_public_dict() for hit in hits],
                    "confirmed_frame": session.confirmed_frame,
                },
            },
        }

    def _cast(self, session: SessionState, casting_prompt: str) -> dict:
        frame = self._confirmed_or_top_frame(session)
        if frame is None:
            return self._search(session, casting_prompt)

        result = self.cast_provider.cast_into_frame(frame.record.frame_path, casting_prompt)
        session.casting_prompt = casting_prompt
        session.anchor_frames = [result["frame_path"]]

        return self._cast_response(result, casting_prompt)

    @staticmethod
    def _cast_response(result: dict, casting_prompt: str) -> dict:
        return {
            "reply": "I made an anchor frame from that match. Approve it when it feels right, or ask for a revision.",
            "ui_action": {
                "type": "show_frame_gallery",
                "payload": {
                    "primary": {
                        "frame_id": result["frame_id"],
                        "frame_path": result["frame_path"],
                        "caption": f"Cast frame: {casting_prompt}",
                        "provider": result.get("provider"),
                        "fallback": result.get("fallback", False),
                        "attempts": result.get("attempts"),
                    },
                    "frames": [
                        {
                            "frame_id": result["frame_id"],
                            "frame_path": result["frame_path"],
                            "caption": f"Cast frame: {casting_prompt}",
                            "score": 1.0,
                            "provider": result.get("provider"),
                            "fallback": result.get("fallback", False),
                            "attempts": result.get("attempts"),
                        }
                    ],
                },
            },
        }

    def _synthesize(self, session: SessionState, message: str) -> dict:
        if not session.anchor_frames:
            return {
                "reply": "I need an approved anchor frame first. Tell me what to cast into the matched frame.",
                "ui_action": {"type": "none", "payload": {}},
            }

        prior = session.video_history[-1]["video_id"] if session.video_history else None
        edit_instruction = message if prior and REVISION_RE.search(message) else None
        narrative_hint = self._compose_narrative(session, message)
        return self._synthesize_with_args(
            session,
            narrative_hint=narrative_hint,
            edit_instruction=edit_instruction,
            prior_video_id=prior if edit_instruction else None,
        )

    def _generate_storyboard(self, session: SessionState, story: str, style: str) -> dict:
        if not session.last_hits:
            return {
                "reply": "I need some matched frames to generate a storyboard. Please search for a scene first.",
                "ui_action": {"type": "none", "payload": {}},
            }
            
        storyboard_frames = []
        # Generate storyboard frames using cast_provider for the top 3 hits
        for hit in session.last_hits[:3]:
            # We use the story as the casting prompt for each frame to adapt it to the storyboard
            result = self.cast_provider.cast_into_frame(hit.record.frame_path, story)
            storyboard_frames.append({
                "frame_id": result["frame_id"],
                "frame_path": result["frame_path"],
                "caption": f"Storyboard frame: {story[:30]}...",
                "provider": result.get("provider"),
                "fallback": result.get("fallback", False),
                "attempts": result.get("attempts"),
            })
            
        return {
            "reply": f"I've generated a {style}-style storyboard based on your story using the matching frames.",
            "ui_action": {
                "type": "show_storyboard",
                "payload": {
                    "story": story,
                    "style": style,
                    "frames": storyboard_frames
                }
            }
        }

    def _synthesize_with_args(
        self,
        session: SessionState,
        *,
        narrative_hint: str,
        duration_hint_s: int = 15,
        edit_instruction: str | None = None,
        prior_video_id: str | None = None,
    ) -> dict:
        if not session.anchor_frames:
            return {
                "reply": "I need an approved anchor frame first. Tell me what to cast into the matched frame.",
                "ui_action": {"type": "none", "payload": {}},
            }
        result = self.video_provider.synthesize_video(
            session.anchor_frames,
            narrative_hint,
            duration_hint_s=duration_hint_s,
            edit_instruction=edit_instruction,
            prior_video_id=prior_video_id,
        )
        session.video_history.append(result)

        return {
            "reply": "I composed the approved anchor into a short conversational video preview.",
            "ui_action": {"type": "show_generated_video", "payload": result},
        }

    def _confirmed_or_top_frame(self, session: SessionState) -> SearchHit | None:
        if not session.last_hits:
            return None
        if session.confirmed_frame:
            for hit in session.last_hits:
                if hit.record.id == session.confirmed_frame:
                    return hit
        return session.last_hits[0]

    @staticmethod
    def _should_cast(session: SessionState, message: str) -> bool:
        return bool(session.matched_frames and CAST_RE.search(message) and not APPROVAL_RE.search(message))

    @staticmethod
    def _should_synthesize(session: SessionState, message: str) -> bool:
        return bool(session.anchor_frames and (APPROVAL_RE.search(message) or session.video_history))

    @staticmethod
    def _selection_from_message(session: SessionState, message: str) -> str | None:
        if not session.last_hits:
            return None
        match = SELECTION_RE.search(message)
        if not match:
            return None
        choice = match.group("choice").lower()
        if choice == "last":
            index = len(session.last_hits) - 1
        elif choice in ORDINALS:
            index = ORDINALS[choice]
        elif choice.isdigit():
            index = int(choice) - 1
        else:
            return None
        if 0 <= index < len(session.last_hits):
            return session.last_hits[index].record.id
        return None

    @staticmethod
    def _compose_narrative(session: SessionState, latest_message: str) -> str:
        selected_hit = session.last_hits[0] if session.last_hits else None
        if session.confirmed_frame:
            selected_hit = next(
                (hit for hit in session.last_hits if hit.record.id == session.confirmed_frame),
                selected_hit,
            )
        search_context = selected_hit.record.caption if selected_hit else "the selected local moment"
        cast_context = session.casting_prompt or "the approved subject"
        return (
            f"Use the local moment '{search_context}' as the setup. Feature {cast_context}. "
            f"Keep the pacing conversational with quick montage cuts. Latest user intent: {latest_message}"
        )

    @staticmethod
    def _public_state(session: SessionState) -> dict:
        return {
            "matched_frames": session.matched_frames,
            "confirmed_frame": session.confirmed_frame,
            "anchor_frames": session.anchor_frames,
            "video_history": session.video_history,
            "workflow_stage": AgentOrchestrator._workflow_stage(session),
        }

    def public_state(self, session: SessionState) -> dict:
        return self._public_state(session)

    @staticmethod
    def _workflow_stage(session: SessionState) -> str:
        if session.video_history:
            return "video_ready"
        if session.anchor_frames:
            return "anchor_ready"
        if session.confirmed_frame:
            return "frame_confirmed"
        if session.matched_frames:
            return "matches_ready"
        return "idle"
