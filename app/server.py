"""MCP-Server (low-level) für WhisperX, ausgeliefert über Streamable-HTTP mit Bearer-Token."""
from __future__ import annotations

import base64
import contextlib
import json
import logging

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config import get_config
from .formatting import format_for_chat, format_output
from .transcribe import TranscriptionError, get_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("whisperx-mcp")

cfg = get_config()
server = Server("whisperx-mcp")

# WHY(langdock): Die file-Property MUSS JSON-Schema "format": "file" tragen, damit Langdock
# einen hochgeladenen File-Reference zu {fileName, mimeType, base64, size} auflöst.
TRANSCRIBE_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {
            "type": "object",
            "format": "file",
            "description": "Audio-/Videodatei zur Transkription",
            "properties": {
                "fileName": {"type": "string"},
                "mimeType": {"type": "string"},
                "base64": {"type": "string"},
                "size": {"type": "number"},
            },
            "required": ["fileName", "base64"],
        },
        "num_speakers": {
            "type": "number",
            "description": "Exakte Anzahl Sprecher (leer = Auto-Erkennung)",
        },
        "min_speakers": {"type": "number", "description": "Mindestanzahl Sprecher (optional)"},
        "max_speakers": {"type": "number", "description": "Maximalanzahl Sprecher (optional)"},
        "output_format": {
            "type": "string",
            "enum": ["json", "srt", "vtt", "txt"],
            "default": "json",
            "description": "Ausgabeformat (Standard: json als lesbares Transkript)",
        },
    },
    "required": ["file"],
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="transcribe-audio",
            description=(
                "Transkribiert eine Audio- oder Videodatei mit WhisperX. "
                "Erkennt automatisch Sprecher (Diarization) und liefert Word-Timestamps. "
                "Unterstützt: mp3, wav, m4a, ogg, flac, mp4, webm."
            ),
            inputSchema=TRANSCRIBE_INPUT_SCHEMA,
        )
    ]


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    if name != "transcribe-audio":
        raise ValueError(f"Unbekanntes Tool: {name}")

    file_obj = arguments.get("file") or {}
    b64 = file_obj.get("base64")
    if not b64:
        raise ValueError("Kein 'file.base64'-Inhalt übergeben.")
    filename = file_obj.get("fileName", "audio")
    output_format = arguments.get("output_format") or "json"

    try:
        audio_bytes = base64.b64decode(b64)
    except Exception as exc:
        raise ValueError(f"file.base64 ist kein gültiges Base64: {exc}") from exc

    pipeline = get_pipeline()

    def _run() -> dict:
        return pipeline.transcribe(
            audio_bytes,
            filename=filename,
            num_speakers=_coerce_int(arguments.get("num_speakers")),
            min_speakers=_coerce_int(arguments.get("min_speakers")),
            max_speakers=_coerce_int(arguments.get("max_speakers")),
        )

    try:
        result = await anyio.to_thread.run_sync(_run)
    except TranscriptionError as exc:
        return [types.TextContent(type="text", text=f"Fehler bei der Transkription: {exc}")]

    if output_format == "json":
        text = format_for_chat(result)
    else:
        formatted = format_output(result, output_format)
        text = formatted if isinstance(formatted, str) else json.dumps(formatted, ensure_ascii=False)

    return [types.TextContent(type="text", text=text)]


# ---- HTTP transport + auth -------------------------------------------------------
session_manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)


async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
    await session_manager.handle_request(scope, receive, send)


class BearerAuthMiddleware:
    """Schützt /mcp mit dem konfigurierten API_TOKEN. /health bleibt offen."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/mcp"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        presented = auth[7:] if auth.lower().startswith("bearer ") else auth
        if presented != self.token:
            await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
            return
        await self.app(scope, receive, send)


async def health(_: Request) -> JSONResponse:
    pipeline = get_pipeline()
    return JSONResponse(
        {
            "status": "ok" if pipeline.model_loaded else "loading",
            "gpu": pipeline.gpu_name,
            "model_loaded": pipeline.model_loaded,
            "model": cfg.model_name,
        }
    )


async def root(_: Request) -> PlainTextResponse:
    return PlainTextResponse("whisperx-mcp — MCP endpoint at /mcp")


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    cfg.require_api_token()
    logger.info("Lade WhisperX-Modelle (%s, %s/%s)…", cfg.model_name, cfg.device, cfg.compute_type)
    await anyio.to_thread.run_sync(get_pipeline().load)
    logger.info("Modelle geladen. Diarization=%s", cfg.enable_diarization)
    async with session_manager.run():
        yield


app = Starlette(
    routes=[
        Route("/", root),
        Route("/health", health),
        Mount("/mcp", app=handle_streamable_http),
    ],
    lifespan=lifespan,
)
app.add_middleware(BearerAuthMiddleware, token=cfg.require_api_token())
