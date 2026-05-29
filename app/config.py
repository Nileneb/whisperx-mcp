import os
from functools import lru_cache
from pathlib import Path


def _read_vocab_prompt() -> str:
    explicit = os.environ.get("VOCAB_PROMPT")
    if explicit:
        return explicit
    vocab_file = os.environ.get("VOCAB_FILE", "vocab/altenhilfe.txt")
    path = Path(vocab_file)
    if path.is_file():
        terms = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return ", ".join(terms)
    return ""


class Config:
    def __init__(self) -> None:
        # WHY: ohne API_TOKEN wäre der Service offen erreichbar — harter Abbruch statt stillem Default.
        self.api_token = os.environ.get("API_TOKEN", "").strip()
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()

        self.model_name = os.environ.get("MODEL", "large-v3-turbo")
        self.device = os.environ.get("DEVICE", "cuda")
        self.compute_type = os.environ.get("COMPUTE_TYPE", "float16" if self.device == "cuda" else "int8")
        self.default_language = os.environ.get("LANGUAGE", "de")
        self.batch_size = int(os.environ.get("BATCH_SIZE", "8"))
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", "1") not in ("0", "false", "False")
        # WHY: pyannote.audio 4.x (von whisperx>=3.4 gezogen) lädt für die Diarization immer
        # Embedding-Artefakte aus speaker-diarization-community-1 — dieses Repo muss freigeschaltet sein.
        self.diarize_model = os.environ.get("DIARIZE_MODEL", "pyannote/speaker-diarization-community-1")
        self.vocab_prompt = _read_vocab_prompt()

    def require_api_token(self) -> str:
        if not self.api_token:
            raise RuntimeError("API_TOKEN ist nicht gesetzt — Service würde ungeschützt laufen.")
        return self.api_token


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
