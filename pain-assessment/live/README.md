# Pain Mapper — Claude-driven mode

The offline app one directory up runs a fixed decision tree on your phone. This mode hands
the interview to Claude: it reads everything you've answered so far, decides what to ask
next, and writes the final assessment and massage plan itself.

It talks to Claude through the **`claude` CLI on your subscription — no API key, no
metered billing.**

```
phone browser  ──http──▶  server.mjs  ──spawn──▶  claude -p  ──▶  your subscription
```

The CLI has to run somewhere that isn't a phone, so `server.mjs` lives on a cloud box (or
any always-on machine) and serves the UI to your phone over the network.

## Setting it up

On the box, once:

```bash
# 1. Node 18+ and the Claude CLI
npm install -g @anthropic-ai/claude-code

# 2. log in — this is what makes it subscription-billed rather than API-billed
claude login
claude -p 'say ok'          # confirm it answers without asking for a key

# 3. get the code
git clone <this repo> && cd pain-assessment/live
```

Then run it:

```bash
node server.mjs
```

It prints the URL to open, with an access token in it:

```
  Open on your phone:  http://10.0.0.4:8787/?t=3f9a2b71
```

To keep it up between sessions, use whatever the box already has — `pm2 start server.mjs`,
a systemd unit, `tmux`, or a container. Nothing about the server needs to survive a restart:
sessions are written to `data/` and reloaded automatically.

### Reaching it from a phone

- **Same network** (home Wi-Fi, tailnet): the printed URL works as-is.
- **Cloud box with a public IP:** put it behind a reverse proxy with TLS. Do not expose
  port 8787 directly — the token is in the query string, so plain HTTP leaks it to anything
  watching the network.
- **Quick and private:** `cloudflared tunnel --url http://localhost:8787` or
  `tailscale serve 8787` gives you an HTTPS URL without opening a firewall port.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `PORT` | `8787` | Listen port |
| `HOST` | `0.0.0.0` | Bind address — set `127.0.0.1` behind a proxy |
| `PAIN_TOKEN` | random each boot | Access token. Set it to keep URLs stable |
| `PAIN_MODEL` | `claude-sonnet-5` | Model for planning and the report |
| `PAIN_BATCH` | `6` | Questions Claude plans per call — see below |
| `PAIN_TIMEOUT_MS` | `150000` | Per-call timeout |
| `PAIN_MAX_CONCURRENT` | `2` | Concurrent CLI processes |
| `PAIN_DATA_DIR` | `./data` | Session storage |
| `CLAUDE_BIN` | `claude` | Path to the CLI |

## How it keeps the interview responsive

A `claude -p` call takes 25-80 seconds. Asking Claude for one question at a time would mean
waiting that long between every tap, which makes the assessment unusable.

Instead **Claude plans a batch of questions at a time** (`PAIN_BATCH`, default 6) and the
server hands them out instantly — a tap inside a batch returns in about 20 **milliseconds**.
Claude still authors every question with the full transcript in front of it; it just looks a
few moves ahead. When the buffer drops below three, a refill starts in the background while
you're still working through what you have.

Claude marks each question with `replanIf` — `high`, `low`, `always` or `never` — flagging
where an answer should make it stop and rethink. When one of those fires, or when you write
a note, the server keeps the next queued question, throws away the rest of the batch, and
re-plans **in the background**. You keep moving; the new plan lands behind you.

That async behaviour matters more than it sounds. Re-planning synchronously — the obvious
implementation — turned out to stall the interview on roughly half of all answers, because
the model marks nearly every question as a branch point.

Measured on a cloud box, sonnet-5: **~25-80 s per planning call, ~$0.02-0.03 per question,
roughly $0.60-0.90 of subscription usage for a full 30-question assessment.**

## The clinical brief

Claude gets a system prompt built from `brief.json` — the interviewing rules, style examples,
the pattern vocabulary, and the muscle reference (locations, referral maps, techniques,
cautions) drawn from the standard trigger point literature. It is told to take massage
details from that reference rather than improvising them.

`brief.json` is **generated from the offline app's own data**, so the two modes can't drift:

```bash
node build-brief.mjs      # rewrites brief.json and brief.md from ../index.html
```

Edit the muscle and pattern content in `../index.html` (the `M` and `H` objects) and
regenerate. `brief.md` is the human-readable copy of the same content, for reviewing what
Claude is actually being told.

Only the regions in play are sent on each call — the full reference is ~15k tokens, a typical
call is ~4k.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | the phone UI |
| `GET /offline` | the standalone offline app, as a fallback |
| `GET /health` | liveness, session count, model |
| `POST /api/start` | begin a session |
| `POST /api/answer` | submit a score and optional note, get the next question |
| `POST /api/position` | switch between sitting and lying, re-plans around it |
| `POST /api/finish` | end early and produce the report |

All `/api/*` routes need the token, as `?t=` or `Authorization: Bearer`.

## If something breaks

- **"bad or missing token"** — open the exact URL the server printed.
- **"claude exited 1"** — the CLI isn't logged in on the box. Run `claude login`, then
  `claude -p 'say ok'` as the same user the server runs as.
- **Timeouts** — the box is slow or the model is busy. Raise `PAIN_TIMEOUT_MS`, or drop
  `PAIN_BATCH` to 4 so each call has less to produce.
- **Everything hangs at the first question** — that first call plans the whole opening
  batch and is the slowest of the session. It should still land inside ~30 s.
- **You lose the network mid-assessment** — the UI offers `/offline`. That's a fresh start,
  not a resume, but it works with no server at all.

## Safety

Claude is instructed to screen for red flags first (bladder/bowel change or saddle numbness,
progressive weakness, fever/night sweats/weight loss, significant trauma, unrelenting night
pain) and to surface them prominently in the report, and it is told never to recommend
pressing hard on the front of the neck, deep into the abdomen near a pulse, on a swollen or
hot calf, or along a nerve already producing electric symptoms.

Because a model writes the questions and the report, both can be wrong in ways the fixed
offline version cannot be — it can invent a plausible-sounding test or overstate confidence.
This is a self-assessment aid, not a diagnosis, and it says so on the results screen.
