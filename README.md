[Инструкция на русском языке здесь](README.ru.md).

# AI news digest for Telegram

A Python script that reads selected public Telegram channels, asks Gemini to select up to ten AI news items, and writes a Russian-language digest. It can also send the digest to a destination you configure. Yesterday's local digest is included for deduplication.

## Setup

Requires Python 3.11 or later and your own Telegram API credentials and Gemini API key.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Fill in `TG_API_ID`, `TG_API_HASH` and `GEMINI_API_KEY`. Obtain Telegram API credentials from [my.telegram.org](https://my.telegram.org) and a Gemini key from [Google AI Studio](https://aistudio.google.com). Replace the placeholder in `sources.yml` with public channel usernames, without `@`. Only `username`/`title`/`enabled` are used; there is no priority-weighted ranking or SQLite storage in this version.

`DIGEST_PEER` is optional: leave it empty for local output, or set your destination channel/chat and make sure your Telegram account can post there. `UTC_OFFSET_HOURS` controls the digest date. The model names in `.env.example` are configurable; availability depends on your API account.

## Run

```powershell
./start_bot.bat
```

The first run asks you to sign into Telegram, including two-factor authentication if enabled. This uses a **Telegram user session**, not a BotFather bot token. Keep the resulting `.session` file private. Each invocation creates one digest; scheduling is external to this project.

Posts and yesterday's digest are sent to Gemini. Review a local digest before enabling automatic delivery. The original brand and multi-link filters are retained, and may omit relevant posts. Requests can incur API costs. Long digests can exceed Telegram's message limit; there is no automatic message splitting.

## Verification and publication

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The public export contains source code and generic configuration only. Original credentials, account sessions, channel selection, generated digests, databases, local paths and runtime files are excluded. There is no imported private Git history.

Offline tests check configuration loading, digest generation with a fake model, saving output and optional delivery. They do not verify live Telegram authentication, Gemini model availability or news accuracy. SDK setup follows the [Google Gen AI documentation](https://googleapis.github.io/python-genai/).

MIT license; dependency licenses remain their authors' responsibility.
