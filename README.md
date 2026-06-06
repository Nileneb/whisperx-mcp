# WhisperX MCP

Selbst gehosteter, DSGVO-konformer **WhisperX-Transkriptionsservice** als **MCP-Tool** (Streamable-HTTP).
Ein kombinierter Python-Service (MCP-Server + WhisperX-Pipeline mit Sprecher-Diarisierung), GPU-beschleunigt,
lokal lauffähig.

> Architektur-Hintergrund: siehe [`whisperx-mcp-architektur.md`](./whisperx-mcp-architektur.md).
> Gegenüber dem Doc bewusst vereinfacht: ein Service statt Node-Bridge + Worker, Token-Auth statt OAuth.

## Voraussetzungen

1. **GPU + Docker** mit `nvidia-container-toolkit` (Docker ≥ 25 nutzt **CDI** — kein Daemon-Restart,
   keine Störung paralleler Container):
   ```bash
   # einmalig auf dem Host:
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   # Der Installer erzeugt die CDI-Spec (/var/run/cdi/nvidia.yaml) automatisch.
   # Falls nicht: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
   docker run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi   # Test
   ```
   Compose nutzt `driver: cdi` / `nvidia.com/gpu=all` (siehe `docker-compose.yml`).
2. **Hugging Face**: Token erzeugen **und** Lizenzen akzeptieren (Gated-Repos, Freigabe i. d. R. sofort):
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>  (Standardmodell, pyannote 4.x)
   - <https://huggingface.co/pyannote/segmentation-3.0>

   Alternativ ein anderes Diarization-Modell via `DIARIZE_MODEL` in `.env` setzen.

## Setup

```bash
cp .env.example .env
# API_TOKEN setzen (schützt /mcp/):
echo "API_TOKEN=$(openssl rand -hex 32)" >> .env   # oder manuell eintragen
# HF_TOKEN in .env eintragen

docker compose up -d --build
docker compose logs -f whisperx-mcp     # wartet auf "Modelle geladen."
```

Der erste Start lädt die Modelle herunter (mehrere GB → `model-cache`-Volume). Der Worker beginnt
erst nach `Modelle geladen.` auf Port **8000** zu lauschen (`/health` liefert dann `model_loaded: true`).

## Endpoints

Der Service lauscht auf **`localhost:8000`**:

| Pfad | Methode | Auth | Zweck |
|---|---|---|---|
| `/health` | GET | — | Liveness/Readiness: `{"status":"ok","ready":true,"model_loaded":true,…}` |
| `/mcp/` | POST | Bearer | MCP-Streamable-HTTP-Endpunkt (Trailing-Slash beachten) |

```bash
curl -s http://localhost:8000/health                                   # {"status":"ok","ready":true,…}
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/mcp/   # 401 (ohne Token)
```

`/health` ist bewusst auth-frei (nur Info) und meldet `ready: true`, sobald der Service lauscht —
unabhängig davon, ob das Modell gerade im VRAM resident ist (der Energy-Saver entlädt es nach Idle).
`model_loaded` zeigt den VRAM-Zustand.

## MCP-Anbindung

Beliebiger MCP-Client (z. B. Langdock → *Custom MCP Integration*):

1. **Endpoint-URL**: `http://localhost:8000/mcp/`  ← **mit Trailing-Slash**.
   Hinter einem TLS-terminierenden Reverse-Proxy entsprechend die öffentliche `https://…/mcp/`-URL.
2. **Auth**: *API Key* — Header `Authorization`, Wert `Bearer <API_TOKEN>`.
3. **Transport**: Streamable HTTP.
4. Nach dem Verbinden erscheint das Tool **`transcribe-audio`**. Audio hochladen → lesbares Transkript.

## Tool: `transcribe-audio`

| Parameter | Typ | Default | Beschreibung |
|---|---|---|---|
| `file` | file | — | Audio/Video (mp3, wav, m4a, ogg, flac, mp4, webm) |
| `num_speakers` | int | auto | exakte Sprecheranzahl |
| `min_speakers` / `max_speakers` | int | — | Grenzen für Auto-Erkennung |
| `output_format` | enum | `json` | `json` (lesbares Transkript) / `srt` / `vtt` / `txt` |

Fachvokabular (initial_prompt) wird aus [`vocab/altenhilfe.txt`](./vocab/altenhilfe.txt) geladen —
pro Sektor erweiterbar oder via `VOCAB_PROMPT`/`VOCAB_FILE` überschreibbar.

## Reverse-Proxy (optional, für externen Zugriff)

Für Zugriff von außen einen TLS-terminierenden Reverse-Proxy (Caddy/nginx/NAS o. Ä.) vor den Service setzen,
der per HTTPS auf `http://<host>:8000` proxyt. Wichtig:
- **Proxy-Timeout hochsetzen** (z. B. 1800 s) — lange Transkriptionen dürfen nicht abgebrochen werden.
- Beim `/mcp`→`/mcp/`-Redirect das Schema (https) erhalten (Proxy `--proxy-headers` reicht der Worker
  bereits durch; siehe `Dockerfile`-CMD).

## Bekannte Limits

- **Lange Dateien**: Verarbeitung dauert Minuten; für Tests kurze Aufnahmen (≲ 15 min) empfehlen.
  Async-Job-Polling (zweites MCP-Tool) ist die spätere Lösung (Doc §5).
- **Parallelität**: 1 Job gleichzeitig (GPU-Lock).
