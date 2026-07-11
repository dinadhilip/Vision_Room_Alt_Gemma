from __future__ import annotations

import base64
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont

from .config import Settings


class CastProvider:
    def cast_into_frame(
        self,
        base_frame_path: str,
        casting_prompt: str,
        reference_image_path: str | None = None,
    ) -> dict:
        raise NotImplementedError


class VideoProvider:
    def synthesize_video(
        self,
        anchor_frame_paths: list[str],
        narrative_hint: str,
        duration_hint_s: int = 15,
        edit_instruction: str | None = None,
        prior_video_id: str | None = None,
    ) -> dict:
        raise NotImplementedError


class DemoCastProvider(CastProvider):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def cast_into_frame(
        self,
        base_frame_path: str,
        casting_prompt: str,
        reference_image_path: str | None = None,
    ) -> dict:
        source = Path(base_frame_path)
        frame_id = f"cast_{uuid.uuid4().hex[:10]}"
        output_path = self.output_dir / f"{frame_id}.png"

        if source.exists():
            image = Image.open(source).convert("RGB")
        else:
            image = Image.new("RGB", (1280, 720), (25, 31, 42))

        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        panel_height = max(110, height // 5)
        draw.rectangle((0, height - panel_height, width, height), fill=(7, 10, 18, 205))
        draw.rectangle((24, 24, width - 24, height - 24), outline=(125, 211, 252, 180), width=4)

        label = f"Cast: {casting_prompt.strip() or 'subject'}"
        if reference_image_path:
            label += " | identity reference locked"
        self._multiline(draw, label, (42, height - panel_height + 28), width - 84)
        image.save(output_path)
        return {"frame_id": frame_id, "frame_path": str(output_path)}

    @staticmethod
    def _multiline(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], max_width: int) -> None:
        font = ImageFont.load_default()
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=font) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        y = xy[1]
        for line in lines[:4]:
            draw.text((xy[0], y), line, fill=(245, 247, 250, 255), font=font)
            y += 22


class DemoVideoProvider(VideoProvider):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_video(
        self,
        anchor_frame_paths: list[str],
        narrative_hint: str,
        duration_hint_s: int = 15,
        edit_instruction: str | None = None,
        prior_video_id: str | None = None,
    ) -> dict:
        video_id = f"video_{uuid.uuid4().hex[:10]}"
        manifest_path = self.output_dir / f"{video_id}.json"
        preview_path = self.output_dir / f"{video_id}.png"
        self._make_preview(preview_path, anchor_frame_paths, narrative_hint, edit_instruction)

        manifest = {
            "video_id": video_id,
            "video_url": f"/assets/generated/{preview_path.name}",
            "manifest_url": f"/assets/generated/{manifest_path.name}",
            "provider": "demo",
            "anchor_frame_paths": anchor_frame_paths,
            "narrative_hint": narrative_hint,
            "duration_hint_s": duration_hint_s,
            "edit_instruction": edit_instruction,
            "prior_video_id": prior_video_id,
        }
        manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
        return {"video_id": video_id, "video_url": manifest["video_url"], "manifest_url": manifest["manifest_url"]}

    def _make_preview(
        self,
        preview_path: Path,
        anchor_frame_paths: list[str],
        narrative_hint: str,
        edit_instruction: str | None,
    ) -> None:
        canvas = Image.new("RGB", (1280, 720), (19, 24, 33))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, 1280, 720), fill=(19, 24, 33, 255))

        for index, path in enumerate(anchor_frame_paths[:3]):
            source = Path(path)
            if source.exists():
                thumb = Image.open(source).convert("RGB")
                thumb.thumbnail((360, 240))
                x = 70 + index * 405
                canvas.paste(thumb, (x, 110))
                draw.rectangle((x, 110, x + thumb.width, 110 + thumb.height), outline=(125, 211, 252, 180), width=3)

        font = ImageFont.load_default()
        title = "Omni Flash preview manifest"
        draw.text((70, 420), title, fill=(245, 247, 250, 255), font=font)
        body = narrative_hint[:220] or "Conversational montage from approved anchor frame."
        if edit_instruction:
            body += f" Edit: {edit_instruction[:160]}"
        y = 460
        for line in _wrap_text(draw, body, font, 1120):
            draw.text((70, y), line, fill=(198, 210, 224, 255), font=font)
            y += 24
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(preview_path)


class HttpCastProvider(CastProvider):
    def __init__(self, endpoint: str, api_key: str | None, fallback: CastProvider) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.fallback = fallback

    def cast_into_frame(
        self,
        base_frame_path: str,
        casting_prompt: str,
        reference_image_path: str | None = None,
    ) -> dict:
        try:
            payload: dict[str, Any] = {
                "base_frame": _read_b64(base_frame_path),
                "casting_prompt": casting_prompt,
                "reference_image": _read_b64(reference_image_path) if reference_image_path else None,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = httpx.post(self.endpoint, json=payload, headers=headers, timeout=90)
            response.raise_for_status()
            return response.json()
        except Exception:
            return self.fallback.cast_into_frame(base_frame_path, casting_prompt, reference_image_path)


class HttpVideoProvider(VideoProvider):
    def __init__(self, endpoint: str, api_key: str | None, fallback: VideoProvider) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.fallback = fallback

    def synthesize_video(
        self,
        anchor_frame_paths: list[str],
        narrative_hint: str,
        duration_hint_s: int = 15,
        edit_instruction: str | None = None,
        prior_video_id: str | None = None,
    ) -> dict:
        try:
            payload = {
                "anchor_frames": [_read_b64(path) for path in anchor_frame_paths],
                "narrative_hint": narrative_hint,
                "duration_hint_s": duration_hint_s,
                "edit_instruction": edit_instruction,
                "prior_video_id": prior_video_id,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = httpx.post(self.endpoint, json=payload, headers=headers, timeout=180)
            response.raise_for_status()
            return response.json()
        except Exception:
            return self.fallback.synthesize_video(
                anchor_frame_paths,
                narrative_hint,
                duration_hint_s,
                edit_instruction,
                prior_video_id,
            )


def build_cast_provider(settings: Settings) -> CastProvider:
    demo = DemoCastProvider(settings.resolved_generated_dir)
    if settings.nb2_lite_endpoint:
        return HttpCastProvider(settings.nb2_lite_endpoint, settings.nb2_lite_api_key, demo)
    return demo


def build_video_provider(settings: Settings) -> VideoProvider:
    demo = DemoVideoProvider(settings.resolved_generated_dir)
    if settings.omni_flash_endpoint:
        return HttpVideoProvider(settings.omni_flash_endpoint, settings.omni_flash_api_key, demo)
    return demo


def copy_frame_into_uploads(source: Path, uploads_dir: Path) -> Path:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    target = uploads_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def _read_b64(path: str | None) -> str | None:
    if not path:
        return None
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

