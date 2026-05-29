"""WhisperX-Pipeline: Modelle einmal warm laden, Jobs seriell auf der GPU verarbeiten."""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import defaultdict

from .config import get_config

# WHY: GPU verarbeitet einen Job zur Zeit (RTX 3060). Lock serialisiert konkurrierende MCP-Calls.
_gpu_lock = threading.Lock()


class TranscriptionError(Exception):
    """Klartext-Fehler, der bis zum MCP-Client durchgereicht wird."""


class Pipeline:
    def __init__(self) -> None:
        self.cfg = get_config()
        self._model = None
        self._align_model = None
        self._align_metadata = None
        self._align_lang = None
        self._diarize = None
        self._gpu_name = None
        self._loaded = False

    # ---- lazy/warm model loading -------------------------------------------------
    def load(self) -> None:
        import torch
        import whisperx

        device = self.cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            raise TranscriptionError(
                "DEVICE=cuda gesetzt, aber CUDA ist im Container nicht verfügbar — "
                "nvidia-container-toolkit / runtime: nvidia prüfen."
            )
        if device == "cuda":
            self._gpu_name = torch.cuda.get_device_name(0)

        asr_options = {}
        if self.cfg.vocab_prompt:
            asr_options["initial_prompt"] = self.cfg.vocab_prompt

        self._model = whisperx.load_model(
            self.cfg.model_name,
            device,
            compute_type=self.cfg.compute_type,
            language=self.cfg.default_language,
            asr_options=asr_options or None,
        )

        if self.cfg.enable_diarization:
            if not self.cfg.hf_token:
                raise TranscriptionError(
                    "ENABLE_DIARIZATION aktiv, aber HF_TOKEN fehlt — pyannote-Modelle benötigen ihn."
                )
            self._diarize = _load_diarize_pipeline(self.cfg.hf_token, device, self.cfg.diarize_model)

        self._loaded = True

    def _ensure_align(self, language: str, device: str):
        import whisperx

        if self._align_model is None or self._align_lang != language:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language, device=device
            )
            self._align_lang = language
        return self._align_model, self._align_metadata

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def gpu_name(self) -> str | None:
        return self._gpu_name

    # ---- transcription -----------------------------------------------------------
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str | None = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict:
        import whisperx

        if not self._loaded:
            raise TranscriptionError("Modelle sind noch nicht geladen.")

        language = language or self.cfg.default_language
        device = self.cfg.device
        suffix = os.path.splitext(filename)[1] or ".bin"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            try:
                audio = whisperx.load_audio(tmp.name)
            except Exception as exc:  # ffmpeg-Decoding fehlgeschlagen → ungültiges Format
                raise TranscriptionError(
                    f"Audio konnte nicht dekodiert werden ({filename}): {exc}. "
                    "Unterstützt: mp3, wav, m4a, ogg, flac, mp4, webm."
                ) from exc

            with _gpu_lock:
                t0 = time.perf_counter()
                try:
                    result = self._model.transcribe(audio, batch_size=self.cfg.batch_size, language=language)

                    align_model, align_meta = self._ensure_align(language, device)
                    result = whisperx.align(
                        result["segments"], align_model, align_meta, audio, device,
                        return_char_alignments=False,
                    )

                    if self._diarize is not None:
                        diarize_segments = self._diarize(
                            audio,
                            num_speakers=num_speakers,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                        )
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        import torch

                        torch.cuda.empty_cache()
                        raise TranscriptionError(
                            "GPU-Speicher erschöpft (CUDA OOM). Datei ist zu groß/lang für die RTX 3060 "
                            "in einem Durchgang — kürzere Aufnahme verwenden."
                        ) from exc
                    raise
                processing_time = round(time.perf_counter() - t0, 1)

        duration = round(float(len(audio)) / 16000.0, 1)  # whisperx lädt mit 16 kHz
        return _build_response(result, filename, language, duration, processing_time)


def _load_diarize_pipeline(hf_token: str, device: str, model_name: str):
    # WHY: Import-Pfad UND Auth-Kwarg der DiarizationPipeline wechselten zwischen whisperx-Versionen
    # (>=3.4: token=, <3.4: use_auth_token=).
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:
        from whisperx import DiarizationPipeline  # type: ignore
    try:
        return DiarizationPipeline(model_name=model_name, token=hf_token, device=device)
    except TypeError:
        return DiarizationPipeline(model_name=model_name, use_auth_token=hf_token, device=device)


def _build_response(result: dict, filename: str, language: str, duration: float, processing_time: float) -> dict:
    raw_segments = result.get("segments", [])
    segments = []
    speaker_time = defaultdict(float)
    speaker_count = defaultdict(int)
    speakers_seen = set()

    for seg in raw_segments:
        start = seg.get("start")
        end = seg.get("end")
        speaker = seg.get("speaker")
        words = [
            {
                "word": w.get("word"),
                "start": w.get("start"),
                "end": w.get("end"),
                **({"speaker": w["speaker"]} if w.get("speaker") else {}),
            }
            for w in seg.get("words", [])
            if w.get("start") is not None
        ]
        segments.append(
            {
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": seg.get("text", "").strip(),
                "words": words,
            }
        )
        if speaker is not None and start is not None and end is not None:
            speakers_seen.add(speaker)
            speaker_time[speaker] += float(end) - float(start)
            speaker_count[speaker] += 1

    speakers = {
        spk: {"total_speaking_time": round(speaker_time[spk], 1), "segment_count": speaker_count[spk]}
        for spk in sorted(speakers_seen)
    }

    return {
        "metadata": {
            "filename": filename,
            "duration_seconds": duration,
            "language": language,
            "num_speakers": len(speakers_seen) if speakers_seen else None,
            "processing_time_seconds": processing_time,
        },
        "segments": segments,
        "speakers": speakers,
    }


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline
