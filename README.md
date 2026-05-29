# WhisperX MCP — Test-Deployment (whisper.linn.games)

Selbst gehosteter, DSGVO-konformer WhisperX-Transkriptionsservice als **MCP-Tool für Langdock**.
Ein kombinierter Python-Service (MCP über Streamable-HTTP + WhisperX-Pipeline), TLS via Caddy,
für eine kleine Testgruppe auf dem eigenen PC (RTX 3060 12 GB).

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
2. **Router**: externes TCP **443 → dieser PC:443** forwarden. (Port 80 unangetastet — host-nginx belegt ihn.)
3. **DNS**: A-Record `whisper.linn.games` → öffentliche Heim-IP (bei dynamischer IP: DynDNS-Updater).
4. **Hugging Face**: Token erzeugen **und** Lizenzen akzeptieren (Gated-Repos, Freigabe i. d. R. sofort):
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>  (Standardmodell, pyannote 4.x)
   - <https://huggingface.co/pyannote/segmentation-3.0>

   Alternativ ein anderes Diarization-Modell via `DIARIZE_MODEL` in `.env` setzen.

## Setup

```bash
cp .env.example .env
# API_TOKEN setzen:
echo "API_TOKEN=$(openssl rand -hex 32)" >> .env   # oder manuell eintragen
# HF_TOKEN in .env eintragen

docker compose up -d --build
docker compose logs -f whisperx-mcp     # wartet auf "Modelle geladen."
```

Der erste Start lädt die Modelle herunter (mehrere GB → `model-cache`-Volume). `start_period` im
Healthcheck ist auf 10 min gesetzt; Caddy startet erst, wenn der Worker `model_loaded: true` meldet.

## Test

```bash
# lokal (Worker direkt, ohne TLS) — temporär Port mappen oder im compose-Netz:
docker compose exec whisperx-mcp python -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://localhost:8000/health')))"

# extern über Caddy:
curl -I https://whisper.linn.games/health          # 200, gültiges Cert
curl https://whisper.linn.games/mcp                 # 401 ohne Token
```

## Langdock-Anbindung

1. In Langdock → Integrations → **Custom MCP Integration**.
2. **Endpoint-URL**: `https://whisper.linn.games/mcp`
3. **Auth**: *API Key Authentication* — Header `Authorization`, Wert `Bearer <API_TOKEN>`.
4. Transport: **Streamable HTTP**.
5. Nach dem Verbinden erscheint das Tool **`transcribe-audio`**. Audio im Chat hochladen → lesbares Transkript.

## Tool: `transcribe-audio`

| Parameter | Typ | Default | Beschreibung |
|---|---|---|---|
| `file` | file | — | Audio/Video (mp3, wav, m4a, ogg, flac, mp4, webm) |
| `num_speakers` | int | auto | exakte Sprecheranzahl |
| `min_speakers` / `max_speakers` | int | — | Grenzen für Auto-Erkennung |
| `output_format` | enum | `json` | `json` (lesbares Transkript) / `srt` / `vtt` / `txt` |

Fachvokabular (initial_prompt) wird aus [`vocab/altenhilfe.txt`](./vocab/altenhilfe.txt) geladen —
pro Sektor erweiterbar oder via `VOCAB_PROMPT`/`VOCAB_FILE` überschreibbar.

## Bekannte Limits (Testphase)

- **Lange Dateien**: Verarbeitung dauert Minuten; für den Test kurze Aufnahmen (≲15 min) empfehlen.
  Async-Job-Polling (zweites MCP-Tool) ist die spätere Lösung (Doc §5).
- **Parallelität**: 1 Job gleichzeitig (GPU-Lock). Reicht für 2–3 Tester.
- **Dynamische Heim-IP**: ohne DynDNS bricht die Erreichbarkeit bei IP-Wechsel.
