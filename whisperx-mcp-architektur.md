# Projekt: WhisperX Transkriptions-Service für die Bergische Diakonie

## Ziel

Ein selbst gehosteter, DSGVO-konformer Spracherkennungsservice mit:
- **Höchste Qualität** für deutsche Sprache (inkl. Dialekte, Fachsprache Pflege/Sozialarbeit)
- **Sprechererkennung** (Diarization) — wer hat wann gesprochen
- **Word-level Timestamps** — exakte Zuordnung jedes Worts
- **Nahtlose Integration** in Langdock via MCP-Tool

---

## 1. Architekturübersicht

```
┌───────────────────────────────────────────────────────────────┐
│                        LANGDOCK                               │
│                                                               │
│   User lädt Audio hoch  ──►  MCP-Tool "transcribe-audio"     │
│                              (format: "file")                 │
│                                    │                          │
│                                    │ base64 FileData          │
└────────────────────────────────────┼──────────────────────────┘
                                     │ HTTPS / OAuth 2.0
                                     ▼
┌───────────────────────────────────────────────────────────────┐
│                   MCP BRIDGE SERVER (Node.js)                 │
│                                                               │
│   • Nimmt FileData entgegen (base64 → Buffer)                │
│   • Authentifizierung via OAuth 2.0                           │
│   • Leitet Audio weiter an WhisperX Worker                   │
│   • Gibt strukturiertes Transkript zurück                    │
│                                                               │
│   Port: 443 (HTTPS)                                          │
└───────────────────────────────┬───────────────────────────────┘
                                │ REST (intern)
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                 WHISPERX WORKER (Python / FastAPI)            │
│                                                               │
│   ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│   │  WhisperX    │  │  pyannote/       │  │  Post-        │  │
│   │  large-v3    │  │  speaker-        │  │  Processing   │  │
│   │  (ASR)       │  │  diarization-3.1 │  │  & Formatierung│ │
│   └──────┬──────┘  └────────┬─────────┘  └───────┬───────┘  │
│          │                  │                     │           │
│          ▼                  ▼                     ▼           │
│   Rohtranskript  →  Sprecher-Labels  →  Fertiges Transkript  │
│                                                               │
│   GPU: NVIDIA (mind. 8 GB VRAM, empf. 16+ GB)               │
│   Port: 8000 (intern, nicht öffentlich)                      │
└───────────────────────────────────────────────────────────────┘
```

### Warum zwei Server?

| Schicht | Sprache | Aufgabe |
|---|---|---|
| **MCP Bridge** | Node.js/TypeScript | MCP-Protokoll, OAuth, File-Handling, Langdock-Kompatibilität |
| **WhisperX Worker** | Python | GPU-beschleunigte Transkription, ML-Modelle |

Die Trennung entkoppelt das MCP-Protokoll von der ML-Pipeline. Der WhisperX Worker kann unabhängig skaliert, neu gestartet oder aktualisiert werden, ohne die Langdock-Verbindung zu beeinflussen.

---

## 2. Komponenten im Detail

### 2.1 WhisperX Worker (Python / FastAPI)

**Kernmodelle:**

| Modell | Version | Zweck | VRAM |
|---|---|---|---|
| `whisper-large-v3-turbo` | OpenAI | ASR (Speech-to-Text) | ~6 GB |
| `pyannote/speaker-diarization-3.1` | pyannote.audio | Sprechertrennung | ~2 GB |
| `pyannote/segmentation-3.0` | pyannote.audio | VAD + Segmentierung | inkl. |

**API-Endpunkte:**

```
POST /transcribe
  Body: multipart/form-data
    - file: <audio-binary>
    - language: "de" (default)
    - num_speakers: int (optional, Auto-Erkennung wenn leer)
    - min_speakers: int (optional)
    - max_speakers: int (optional)
    - output_format: "json" | "srt" | "vtt" | "txt" (default: "json")

GET /health
  → { "status": "ok", "gpu": "NVIDIA A10", "model_loaded": true }
```

**Response-Format (JSON):**

```json
{
  "metadata": {
    "filename": "fallbesprechung_2026-05-28.mp3",
    "duration_seconds": 1847.3,
    "language": "de",
    "language_confidence": 0.97,
    "num_speakers": 4,
    "processing_time_seconds": 42.1
  },
  "segments": [
    {
      "start": 0.0,
      "end": 3.42,
      "speaker": "SPEAKER_00",
      "text": "Guten Morgen, dann fangen wir mal an mit der Fallbesprechung.",
      "words": [
        { "word": "Guten", "start": 0.0, "end": 0.38 },
        { "word": "Morgen,", "start": 0.42, "end": 0.88 },
        { "word": "dann", "start": 1.02, "end": 1.24 }
      ]
    },
    {
      "start": 3.8,
      "end": 8.12,
      "speaker": "SPEAKER_01",
      "text": "Ja, ich wollte über Herrn M. sprechen, der ist seit Montag deutlich unruhiger.",
      "words": []
    }
  ],
  "speakers": {
    "SPEAKER_00": { "total_speaking_time": 487.2, "segment_count": 89 },
    "SPEAKER_01": { "total_speaking_time": 623.1, "segment_count": 112 },
    "SPEAKER_02": { "total_speaking_time": 401.8, "segment_count": 67 },
    "SPEAKER_03": { "total_speaking_time": 335.2, "segment_count": 54 }
  }
}
```

**FastAPI-Kernlogik (Pseudocode):**

```python
import whisperx
import torch
from fastapi import FastAPI, UploadFile, Form
from enum import Enum

app = FastAPI()

# Modelle beim Start laden (warm halten)
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

model = whisperx.load_model("large-v3-turbo", device, compute_type=compute_type, language="de")
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token="HF_TOKEN",  # Hugging Face Token für pyannote
    device=device
)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    language: str = Form("de"),
    num_speakers: int | None = Form(None),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    output_format: str = Form("json"),
):
    # 1. Audio laden
    audio = whisperx.load_audio(file)

    # 2. Transkription (Whisper large-v3-turbo)
    result = model.transcribe(audio, batch_size=16, language=language)

    # 3. Alignment (word-level timestamps)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # 4. Diarization (Sprecherzuordnung)
    diarize_segments = diarize_model(
        audio,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # 5. Formatieren & zurückgeben
    return format_output(result, output_format)
```

---

### 2.2 MCP Bridge Server (Node.js / TypeScript)

**Zweck:** Verbindet Langdock (MCP-Protokoll) mit dem WhisperX Worker.

**Tool-Registrierung:**

```typescript
import { z } from "zod";

server.registerTool(
  "transcribe-audio",
  {
    description:
      "Transkribiert eine Audio- oder Videodatei mit WhisperX. " +
      "Erkennt automatisch Sprecher und liefert Timestamps. " +
      "Unterstützt: mp3, wav, m4a, ogg, flac, mp4, webm.",
    inputSchema: {
      file: z
        .object({
          fileName: z.string(),
          mimeType: z.string(),
          base64: z.string(),
          size: z.number().optional(),
        })
        .describe("Audio-/Videodatei zur Transkription")
        .meta({ format: "file" }),

      num_speakers: z
        .number()
        .optional()
        .describe("Exakte Anzahl Sprecher (leer = Auto-Erkennung)"),

      output_format: z
        .enum(["json", "srt", "vtt", "txt"])
        .default("json")
        .describe("Ausgabeformat"),
    },
  },
  async ({ file, num_speakers, output_format }) => {
    // base64 → Buffer → multipart POST an WhisperX Worker
    const buffer = Buffer.from(file.base64, "base64");
    const form = new FormData();
    form.append("file", new Blob([buffer]), file.fileName);
    form.append("language", "de");
    if (num_speakers) form.append("num_speakers", String(num_speakers));
    form.append("output_format", output_format ?? "json");

    const response = await fetch("http://whisperx-worker:8000/transcribe", {
      method: "POST",
      body: form,
    });

    const result = await response.json();

    // Für Langdock: Lesbares Transkript als Text zurückgeben
    const readableTranscript = formatForChat(result);

    return {
      content: [{ type: "text", text: readableTranscript }],
    };
  }
);
```

---

### 2.3 Authentifizierung (OAuth 2.0)

```
Langdock  ──►  MCP Bridge Server
                │
                ├─ OAuth 2.0 Authorization Code Flow
                ├─ Token-Endpoint auf eurem Server
                └─ Scopes: transcribe:read, transcribe:write
```

**Einfachere Alternative für den Start:** API-Key-basierte Auth (Header `X-API-Key`), die in Langdock als Custom Integration konfiguriert wird. OAuth kann nachgerüstet werden.

---

## 3. Infrastruktur & Hardware

### Option A: Eigener Server / VM

| Komponente | Minimum | Empfohlen |
|---|---|---|
| **GPU** | NVIDIA T4 (16 GB VRAM) | NVIDIA A10 (24 GB) oder L4 (24 GB) |
| **CPU** | 8 Cores | 16 Cores |
| **RAM** | 32 GB | 64 GB |
| **Storage** | 100 GB SSD | 250 GB NVMe |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **CUDA** | 12.1+ | 12.4 |

### Option B: Cloud GPU (Hetzner / netcup / OVH — EU-hosted, DSGVO)

| Anbieter | GPU-Option | Preis ca. | Standort |
|---|---|---|---|
| **Hetzner** | EX44 + GPU Addon | ab ~120€/Monat | DE (Falkenstein/Nürnberg) |
| **netcup** | Root-Server + GPU | ab ~100€/Monat | DE (Karlsruhe) |
| **OVH** | GPU Instances | ab ~150€/Monat | DE/FR |

### Deployment: Docker Compose

```yaml
version: "3.9"

services:
  whisperx-worker:
    build: ./whisperx-worker
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_TOKEN=${HF_TOKEN}
    volumes:
      - model-cache:/root/.cache
    ports:
      - "127.0.0.1:8000:8000"    # nur intern erreichbar
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mcp-bridge:
    build: ./mcp-bridge
    ports:
      - "443:443"
    environment:
      - WHISPERX_URL=http://whisperx-worker:8000
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - SSL_CERT_PATH=/certs/fullchain.pem
      - SSL_KEY_PATH=/certs/privkey.pem
    volumes:
      - ./certs:/certs:ro
    depends_on:
      whisperx-worker:
        condition: service_healthy

  caddy:   # Reverse Proxy + Auto-SSL (Alternative zu manuellen Certs)
    image: caddy:2
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy-data:/data

volumes:
  model-cache:
  caddy-data:
```

---

## 4. Umsetzungsplan

### Phase 1 — Foundation (Woche 1–2)

| # | Aufgabe | Deliverable |
|---|---|---|
| 1.1 | GPU-Server beschaffen (Hetzner/netcup oder eigene HW) | Laufender Server mit CUDA + Docker |
| 1.2 | Hugging Face Token beantragen (für pyannote-Modelle) | HF_TOKEN bereit |
| 1.3 | WhisperX Worker aufsetzen (Docker) | `/health` erreichbar, GPU erkannt |
| 1.4 | Erster Transkriptionstest via `curl` | Funktionierendes JSON-Transkript |

### Phase 2 — API & Qualität (Woche 3–4)

| # | Aufgabe | Deliverable |
|---|---|---|
| 2.1 | FastAPI-Endpunkte fertigstellen (alle Parameter) | Vollständige REST API |
| 2.2 | Diarization integrieren + testen | Korrekte Sprechertrennung |
| 2.3 | Output-Formate implementieren (JSON, SRT, VTT, TXT) | Alle Formate funktional |
| 2.4 | Deutsch-spezifische Tests: Dialekt, Fachsprache, Eigennamen | Qualitätsbericht mit WER |
| 2.5 | Dateigrößen-Limits + Chunking für lange Aufnahmen (>1h) | Stabile Verarbeitung bis 3h |

### Phase 3 — MCP-Integration (Woche 5–6)

| # | Aufgabe | Deliverable |
|---|---|---|
| 3.1 | MCP Bridge Server implementieren (Node.js) | Tool-Registrierung steht |
| 3.2 | SSL/TLS + Domain einrichten (z.B. `whisper.bergische-diakonie.de`) | HTTPS aktiv |
| 3.3 | OAuth oder API-Key Auth implementieren | Authentifizierung getestet |
| 3.4 | In Langdock als Custom MCP Integration einbinden | Tool in Langdock sichtbar |
| 3.5 | End-to-End-Test: Audio hochladen in Langdock → Transkript | Vollständiger Flow funktioniert |

### Phase 4 — Produktionsreife (Woche 7–8)

| # | Aufgabe | Deliverable |
|---|---|---|
| 4.1 | Error Handling: Timeouts, ungültige Formate, GPU OOM | Robuste Fehlerbehandlung |
| 4.2 | Queue-System für parallele Anfragen (Redis + Celery oder ähnlich) | Bis zu 5 gleichzeitige Jobs |
| 4.3 | Monitoring: GPU-Auslastung, Verarbeitungszeiten, Fehlerrate | Dashboard / Alerts |
| 4.4 | Automatische Modell-Updates (Docker Image Rebuild) | CI/CD Pipeline |
| 4.5 | Dokumentation für Team: Nutzung in Langdock, Best Practices | Internes Handbuch |

---

## 5. Limitierungen & Lösungen

| Problem | Lösung |
|---|---|
| **Große Dateien (>100 MB)** | Langdock File-Upload hat Limits → ggf. direkten Upload-Endpoint am Worker für große Dateien |
| **Lange Verarbeitung (>5 Min)** | Async-Processing: Job-ID zurückgeben → Status-Polling via zweites MCP-Tool |
| **Sprechernamen unbekannt** | SPEAKER_00/01/02 → Nutzer kann im Chat nachfragen "Wer ist SPEAKER_00?" → Agent merkt sich Zuordnung |
| **Fachvokabular falsch erkannt** | Custom Vocabulary / Prompt-Parameter in Whisper: `initial_prompt="Fallbesprechung, Pflegeplanung, Barthel-Index..."` |
| **GPU nicht immer ausgelastet** | Auto-Scaling: Server bei Inaktivität herunterfahren (spart Kosten bei Cloud-GPU) |

---

## 6. Fachvokabular-Optimierung für Diakonie-Kontext

WhisperX unterstützt einen `initial_prompt`-Parameter, der das Modell auf erwartetes Vokabular einstellt:

```python
result = model.transcribe(
    audio,
    language="de",
    initial_prompt=(
        "Fallbesprechung, Pflegeplanung, Pflegegrad, Barthel-Index, "
        "Demenz, Dekubitus, Mobilisation, Biografiearbeit, "
        "Bezugspflege, Dokumentation, MDK, Wundversorgung, "
        "Sozialtherapie, Eingliederungshilfe, Teilhabeplan, "
        "Bergische Diakonie, Altenhilfe, Jugendhilfe"
    ),
)
```

Dieses Prompt-Vokabular kann als konfigurierbare Liste pro Sektor der Bergischen Diakonie gepflegt werden (Altenhilfe vs. Kinder-Jugend-Familie vs. Sozialtherapeutische Hilfe).

---

## 7. Kostenübersicht (geschätzt)

| Posten | Einmalig | Monatlich |
|---|---|---|
| GPU-Server (Hetzner/netcup) | — | 100–150 € |
| Domain + SSL | 10 € | — |
| Entwicklungszeit (intern, ~160h) | 160h × Stundensatz | — |
| Hugging Face Token | kostenlos | — |
| **Laufende Kosten gesamt** | | **~100–150 €/Monat** |

Zum Vergleich: Cloud-APIs (AssemblyAI, Azure) kosten ca. 0,60–1,20 € pro Audiostunde. Ab ~150 Audiostunden/Monat ist Self-Hosting günstiger.

---

## 8. Spätere Erweiterungen (nach Projektabschluss)

- **Anbindung an Diakonie-DB**: Transkripte automatisch dem richtigen Sektor/Einrichtung zuordnen
- **Zusammenfassung**: Langdock Agent erstellt aus Transkript automatisch Protokoll + To-Dos
- **Echtzeit-Transkription**: WebSocket-Stream statt Datei-Upload (z.B. für Live-Meetings)
- **Sprecher-Profile**: Wiederkehrende Sprecher über Sessions hinweg erkennen
- **Fine-Tuning**: Whisper auf Diakonie-spezifische Aufnahmen nachtrainieren (wenn genug Daten)
