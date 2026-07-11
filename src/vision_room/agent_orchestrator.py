from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field

from .providers import CastProvider, VideoProvider
from .local_agent import LocalGemmaPlanner, PlannedReply, PlannedToolCall
from .search_tool import SearchHit, VideoSearchTool


SYSTEM_PROMPT = """You are a conversational video assistant. Use search_video_library for local footage search, cast_into_frame after a confirmed frame, and synthesize_video after an approved anchor frame. Keep replies short and never expose tool JSON to the user."""

APPROVAL_RE = re.compile(r"\b(approve|approved|looks good|use this|go ahead|make video|synthesize|render)\b", re.I)
REVISION_RE = re.compile(r"\b(change|revise|instead|make it|swap|faster|slower|edit)\b", re.I)
CAST_RE = re.compile(r"\b(cast|put|place|add|feature|with|wearing|holding|product|character)\b", re.I)


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


class AgentOrchestrator:
    """Rule-based agent loop with the same tool contract the Gemma loop will use."""

    def __init__(
        self,
        search_tool: VideoSearchTool,
        cast_provider: CastProvider,
        video_provider: VideoProvider,
        *,
        search_confidence_threshold: float = 0.18,
        local_planner: LocalGemmaPlanner | None = None,
    ) -> None:
        self.search_tool = search_tool
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
        return None

    def _search(self, session: SessionState, query: str, top_k: int = 3) -> dict:
        hits = self.search_tool.search_video_library(query, top_k=top_k)
        session.last_hits = hits
        session.matched_frames = [hit.record.id for hit in hits]
        session.confirmed_frame = hits[0].record.id if hits else None

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
                    },
                    "frames": [
                        {
                            "frame_id": result["frame_id"],
                            "frame_path": result["frame_path"],
                            "caption": f"Cast frame: {casting_prompt}",
                            "score": 1.0,
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
    def _compose_narrative(session: SessionState, latest_message: str) -> str:
        search_context = session.last_hits[0].record.caption if session.last_hits else "the selected local moment"
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
        }
