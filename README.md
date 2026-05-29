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
2. **Netzwerk**: Eine **Synology-NAS** ist der öffentliche TLS-Endpunkt (Reverse Proxy).
   Router forwardet **443 → NAS** (und **80 → NAS** für den Let's-Encrypt-ACME-Check). Siehe
   §"Synology Reverse Proxy" unten. Der WhisperX-Worker läuft per HTTP auf `<PC-LAN-IP>:8000`.
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

Der erste Start lädt die Modelle herunter (mehrere GB → `model-cache`-Volume). Der Worker beginnt
erst nach `Modelle geladen.` auf Port 8000 zu lauschen (`/health` liefert dann `model_loaded: true`).

## Synology Reverse Proxy

Die NAS terminiert TLS und proxyt per HTTP auf den PC.

1. **Zertifikat** (DSM → Systemsteuerung → Sicherheit → Zertifikat → Hinzufügen →
   *Let's Encrypt*): Domainname `whisper.linn.games`. Voraussetzung: DNS zeigt auf die NAS und
   **Port 80 ist auf die NAS geforwardet** (ACME HTTP-01).
2. **Reverse-Proxy-Regel** (DSM → Systemsteuerung → Anmeldeportal → Erweitert → Reverse Proxy → Erstellen):
   - Quelle: HTTPS · Hostname `whisper.linn.games` · Port `443`
   - Ziel: HTTP · Hostname `192.168.178.11` (PC-LAN-IP) · Port `8000`
   - Erweiterte Einstellungen → **Proxy-Timeout hochsetzen** (z. B. 1800 s) — sonst bricht die NAS
     lange Transkriptionen ab.
3. **Zertifikat zuordnen** (DSM → Sicherheit → Zertifikat → Einstellungen): dem Dienst
   `whisper.linn.games` das neue Let's-Encrypt-Cert zuweisen.

## Test

```bash
# lokal auf dem PC (Worker direkt):
curl -s http://192.168.178.11:8000/health          # {"status":"ok","model_loaded":true,...}
curl -s -o /dev/null -w '%{http_code}\n' http://192.168.178.11:8000/mcp   # 401 (ohne Token)

# extern über die NAS:
curl -I https://whisper.linn.games/health          # 200, gültiges Cert (kein Hostname-Mismatch)
curl -s -o /dev/null -w '%{http_code}\n' https://whisper.linn.games/mcp   # 401 ohne Token
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
