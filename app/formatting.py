"""Wandelt das WhisperX-Pipeline-Ergebnis in die Ausgabeformate des Architektur-Docs."""
from __future__ import annotations


def _ts(seconds: float, comma: bool = True) -> str:
    if seconds is None:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    h, millis = divmod(millis, 3600_000)
    m, millis = divmod(millis, 60_000)
    s, millis = divmod(millis, 1000)
    sep = "," if comma else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{millis:03d}"


def to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{_ts(seg.get('start'))} --> {_ts(seg.get('end'))}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker:
            text = f"<v {speaker}>{text}"
        lines.append(f"{_ts(seg.get('start'), comma=False)} --> {_ts(seg.get('end'), comma=False)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_txt(segments: list[dict]) -> str:
    out = []
    last_speaker = None
    for seg in segments:
        speaker = seg.get("speaker")
        text = seg.get("text", "").strip()
        if speaker and speaker != last_speaker:
            out.append(f"\n{speaker}:")
            last_speaker = speaker
        out.append(text)
    return " ".join(out).strip() + "\n"


def format_output(result: dict, output_format: str):
    """Gibt das Ergebnis im gewünschten Format zurück (json -> dict, sonst str)."""
    segments = result.get("segments", [])
    if output_format == "srt":
        return to_srt(segments)
    if output_format == "vtt":
        return to_vtt(segments)
    if output_format == "txt":
        return to_txt(segments)
    return result


def format_for_chat(result: dict) -> str:
    """Kompaktes, lesbares Transkript für die Langdock-Text-Antwort."""
    meta = result.get("metadata", {})
    segments = result.get("segments", [])
    header = []
    fn = meta.get("filename")
    if fn:
        header.append(f"Transkript: {fn}")
    dur = meta.get("duration_seconds")
    if dur:
        mins = int(dur // 60)
        secs = int(dur % 60)
        header.append(f"Dauer: {mins}:{secs:02d}")
    n_spk = meta.get("num_speakers")
    if n_spk:
        header.append(f"Sprecher: {n_spk}")
    lang = meta.get("language")
    if lang:
        header.append(f"Sprache: {lang}")

    body = []
    last_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = _ts(seg.get("start"), comma=False)[:-4]  # HH:MM:SS
        if speaker and speaker != last_speaker:
            body.append(f"\n[{start}] {speaker}:")
            last_speaker = speaker
            body.append(text)
        else:
            body.append(text)

    parts = []
    if header:
        parts.append(" · ".join(header))
        parts.append("")
    parts.append(" ".join(body).strip())
    return "\n".join(parts).strip()
