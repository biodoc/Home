/* =============================================================================
   Pain Mapper — Claude-driven mode

   A small bridge that runs wherever the Claude CLI is installed and logged in.
   Claude chooses every question; this server just holds the session, shells out
   to the CLI, and serves the phone UI.

   No API key: everything goes through `claude -p`, on your subscription.

     node server.mjs                  # prints the URL + token to open on a phone
   ============================================================================= */
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { networkInterfaces } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const CFG = {
  port:    +(process.env.PORT || 8787),
  host:    process.env.HOST || '0.0.0.0',
  token:   process.env.PAIN_TOKEN || randomUUID().slice(0, 8),
  bin:     process.env.CLAUDE_BIN || 'claude',
  model:   process.env.PAIN_MODEL || 'claude-sonnet-5',
  batch:   +(process.env.PAIN_BATCH || 8),
  timeout: +(process.env.PAIN_TIMEOUT_MS || 150_000),
  dataDir: process.env.PAIN_DATA_DIR || join(HERE, 'data'),
  maxConcurrent: +(process.env.PAIN_MAX_CONCURRENT || 2)
};
mkdirSync(CFG.dataDir, { recursive: true });
const SCRATCH = join(CFG.dataDir, 'scratch');   // empty cwd: no CLAUDE.md pickup
mkdirSync(SCRATCH, { recursive: true });

const BRIEF = JSON.parse(readFileSync(join(HERE, 'brief.json'), 'utf8'));

/* ---------------------------------------------------------------- claude ---- */
let running = 0;
const waiting = [];
const slot = () => running < CFG.maxConcurrent
  ? (running++, Promise.resolve())
  : new Promise(r => waiting.push(r)).then(() => { running++; });
const release = () => { running--; const n = waiting.shift(); if (n) n(); };

/* The CLI inherits an ambient session when it is launched from inside another
   Claude session — which silently drags that whole conversation into context
   (25k+ tokens, 6x the cost). Strip those so behaviour is identical whether the
   server was started by hand or from an agent. */
const CHILD_ENV = (() => {
  const e = { ...process.env };
  for (const k of ['CLAUDE_CODE_SESSION_ID', 'CLAUDE_SESSION_ID', 'CLAUDECODE',
                   'CLAUDE_CODE_CHILD_SESSION', 'CLAUDE_PID', 'CLAUDE_EFFORT',
                   'CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD']) delete e[k];
  return e;
})();

function callClaude(system, user) {
  return slot().then(() => new Promise((resolve, reject) => {
    const args = [
      '-p', user,
      '--system-prompt', system,
      '--output-format', 'json',
      '--model', CFG.model,
      '--session-id', randomUUID(),
      '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
      '--setting-sources', '',
      '--exclude-dynamic-system-prompt-sections',
      '--allowed-tools', ''
    ];
    const t0 = Date.now();
    const child = spawn(CFG.bin, args, { cwd: SCRATCH, env: CHILD_ENV });
    let out = '', err = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(new Error('Claude timed out')); }, CFG.timeout);
    child.stdout.on('data', d => out += d);
    child.stderr.on('data', d => err += d);
    child.on('error', e => { clearTimeout(timer); release(); reject(e); });
    child.on('close', code => {
      clearTimeout(timer); release();
      if (code !== 0) return reject(new Error(`claude exited ${code}: ${err.slice(0, 400)}`));
      let env;
      try { env = JSON.parse(out); }
      catch { return reject(new Error(`unparseable CLI envelope: ${out.slice(0, 300)}`)); }
      if (env.is_error) return reject(new Error(env.result || 'claude reported an error'));
      resolve({ text: env.result || '', cost: env.total_cost_usd || 0, ms: Date.now() - t0 });
    });
  }));
}

/** Models wrap JSON in prose or fences often enough to be worth handling. */
function extractJSON(text) {
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = fence ? fence[1] : text;
  const s = body.indexOf('{'), e = body.lastIndexOf('}');
  if (s === -1 || e === -1) throw new Error('no JSON object found');
  return JSON.parse(body.slice(s, e + 1));
}

async function askJSON(system, user, { repair = true } = {}) {
  const res = await callClaude(system, user);
  try { return { data: extractJSON(res.text), cost: res.cost, ms: res.ms }; }
  catch (e) {
    if (!repair) throw new Error(`Claude did not return usable JSON: ${e.message}`);
    const fix = await callClaude(
      system,
      `${user}\n\nYour previous reply could not be parsed as JSON (${e.message}). ` +
      `Reply again with ONLY the JSON object — no prose, no code fence, nothing else.`
    );
    return { data: extractJSON(fix.text), cost: res.cost + fix.cost, ms: res.ms + fix.ms };
  }
}

/* ---------------------------------------------------------------- prompts --- */
const CONTRACT_PLAN = `
## Your output

Reply with ONLY a JSON object, no prose and no code fence:

{
  "thinking": "one sentence on what you are trying to rule in or out right now",
  "phase": "screen" | "scan" | "differentiate" | "confirm" | "done",
  "estimateRemaining": <integer, your honest estimate of questions still to come>,
  "questions": [
    {
      "id": "short-kebab-id",
      "region": "neck|shoulder|midback|lowback|hip|knee|foot|arm|jaw|general",
      "side": "L" | "R" | null,
      "title": "3-5 word imperative, e.g. Bend forward",
      "cue": "the instruction, 1-3 sentences, second person, concrete",
      "note": "what to pay attention to, or empty string",
      "warn": "safety caveat if this test has one, or empty string",
      "labels": ["what 0 means", "what 4 means"],
      "needs": "sit" | "lie" | "any",
      "replanIf": "high" | "low" | "always" | "never"
    }
  ]
}

Plan up to QUESTION_BUDGET questions ahead, in the order you want them asked. They are
served to the person one at a time without waiting for you, so only queue questions whose
value does not depend on how the earlier ones in the batch are answered.

"replanIf" is how you keep control: set "high" when a score of 2+ on that question should
make you rethink before the rest of the batch is used, "low" when a score of 0-1 should,
"always" for a genuine fork in the road, "never" when the following questions stand
regardless. Be honest with it — it is the difference between an interview that adapts and
one that reads off a list.

Never re-ask something already covered, even reworded — you are given the list, and a
repeat wastes the one thing they are spending: patience. Each question must earn its place
by ruling something in or out that the previous answers left open.

Set "phase":"done" with an empty questions array when you have enough to report. Do not
pad the interview to look thorough.`.trim();

const CONTRACT_REPORT = `
## Your output

Reply with ONLY a JSON object, no prose and no code fence:

{
  "summary": "2-4 sentences in plain second person: what you think is going on and how confident you are",
  "flags": ["any safety issue that needs a clinician, phrased directly; empty array if none"],
  "regions": { "lowback": 4, "hip": 2 },
  "patterns": [
    { "name": "human readable", "confidence": <0-100>, "why": "the specific answers that point here, cite them",
      "helps": ["..."], "avoid": ["..."] }
  ],
  "muscles": [
    { "name": "...", "confirmed": <true if pressing it reproduced their pain>,
      "referred": <true if pressing sent pain elsewhere>,
      "where": "how to find it by touch", "refers": "where it sends pain",
      "how": "how to work it", "dose": "how long and how often",
      "stretch": "what to do straight after", "caution": "or empty string" }
  ],
  "routine": [ { "step": "Press: ...", "detail": "..." } ],
  "seeSomeone": "when to stop self-treating and get assessed"
}

Rank patterns by what the answers actually support and say so in "why" — cite the real
scores. Never report a pattern for a region that was not tested. Put muscles whose
pressure test reproduced their pain first, and mark them confirmed. Take the muscle
details from the reference above rather than improvising them. 6-9 muscles at most, and
a routine of 5-8 steps.`.trim();

function systemFor(session, kind) {
  const areas = new Set(session.turns.map(t => t.region).filter(Boolean));
  // once we are past the scan, only carry the reference for regions in play
  const scanning = session.turns.length < 10 || areas.size === 0;
  const wanted = scanning ? [] : [...areas];
  const parts = [BRIEF.core, '', '### Examples of the register to match', '', BRIEF.examples];
  if (wanted.length) {
    parts.push('', '## Patterns to reason toward', '');
    for (const a of wanted) if (BRIEF.patternsByArea[a]) parts.push(BRIEF.patternsByArea[a]);
    parts.push('', '## Muscle reference — referral patterns and self-treatment', '',
               'Referral patterns follow the standard trigger point maps. Use these; do not invent locations or techniques.', '');
    for (const a of wanted) if (BRIEF.musclesByArea[a]) parts.push(BRIEF.musclesByArea[a]);
  }
  if (BRIEF.patternsByArea.general) parts.push('', BRIEF.patternsByArea.general);
  parts.push('', BRIEF.boundaries, '',
    kind === 'report' ? CONTRACT_REPORT : CONTRACT_PLAN.replace('QUESTION_BUDGET', String(CFG.batch)));
  return parts.join('\n');
}

function transcript(session) {
  if (!session.turns.length) return '(nothing asked yet)';
  return session.turns.map(t => {
    const side = t.side ? ` (${t.side === 'L' ? 'left' : 'right'})` : '';
    const lab = t.labels ? ` [0=${t.labels[0]}, 4=${t.labels[1]}]` : '';
    const skipped = t.score === null ? ' — SKIPPED' : '';
    const note = t.note ? `\n     THEIR NOTE: "${t.note}"` : '';
    return `- ${t.title}${side}: ${t.score ?? '-'}/4${lab}${skipped}\n     (asked: ${t.cue})${note}`;
  }).join('\n');
}

const stateLine = s =>
  `Position: ${s.position === 'lie' ? 'lying down' : 'sitting/standing'}. ` +
  `Questions asked so far: ${s.turns.length}.` +
  (s.switchedAt ? ` They changed position after question ${s.switchedAt}.` : '');

function planPrompt(session, reason) {
  const covered = [...session.turns.map(t => t.title), ...session.plan.map(q => q.title)];
  return [
    `## Session so far`, '', stateLine(session), '',
    `## Answers`, '', transcript(session), '',
    covered.length
      ? `## Already asked or already queued — do NOT ask any of these again, in any wording\n\n` +
        covered.map(t => `- ${t}`).join('\n') + '\n'
      : '',
    reason ? `## Why you are being asked to re-plan now\n\n${reason}\n` : '',
    `Give me the next questions.`
  ].join('\n');
}

/* ---------------------------------------------------------------- sessions -- */
const sessions = new Map();
const sessFile = id => join(CFG.dataDir, `session-${id}.json`);
function persist(s) {
  try { writeFileSync(sessFile(s.id), JSON.stringify(s, null, 1)); } catch (e) { log('persist failed', e.message); }
}
function loadSessions() {
  for (const f of readdirSync(CFG.dataDir).filter(f => /^session-.*\.json$/.test(f))) {
    try { const s = JSON.parse(readFileSync(join(CFG.dataDir, f), 'utf8')); sessions.set(s.id, s); } catch {}
  }
  if (sessions.size) log(`restored ${sessions.size} session(s) from disk`);
}

function newSession(position) {
  const s = { id: randomUUID().slice(0, 12), created: Date.now(), position,
              turns: [], plan: [], phase: 'screen', estimate: 30,
              cost: 0, calls: 0, done: false, report: null, planning: null };
  sessions.set(s.id, s);
  return s;
}

/** Fill the queue. Only one plan call in flight per session. */
function refill(session, reason) {
  if (session.planning) return session.planning;
  const job = (async () => {
    const { data, cost, ms } = await askJSON(systemFor(session, 'plan'), planPrompt(session, reason));
    session.cost += cost; session.calls++;
    session.phase = data.phase || session.phase;
    session.estimate = Number.isFinite(data.estimateRemaining) ? data.estimateRemaining : session.estimate;
    const qs = Array.isArray(data.questions) ? data.questions : [];
    const seen = [...session.turns, ...session.plan];
    const fresh = [];
    let dropped = 0;
    for (const raw of qs.filter(valid).map(normalise)) {
      if (isDupe(raw, [...seen, ...fresh])) { dropped++; continue; }
      fresh.push(raw);
    }
    // never let dedupe empty a batch — that stalls the interview waiting on a re-plan
    if (!fresh.length && qs.length) {
      const [first] = qs.filter(valid).map(normalise);
      if (first) { fresh.push(first); dropped--; log(`[${session.id}] dedupe would have emptied the batch — keeping one`); }
    }
    session.plan.push(...fresh);
    if (dropped > 0) log(`[${session.id}] dropped ${dropped} repeat question(s)`);
    log(`[${session.id}] planned ${fresh.length} (${session.phase}) in ${ms}ms $${cost.toFixed(4)}${reason ? ' — ' + reason.slice(0, 60) : ''}`);
    if (data.phase === 'done' && !session.plan.length) session.readyToReport = true;
    persist(session);
  })().finally(() => { session.planning = null; });
  session.planning = job;
  return job;
}

const STOP = new Set(['the','and','for','you','your','any','with','that','this','from','how','are','has','have',
  'rate','does','did','when','what','into','out','off','over','not','but','its','one','all','can','get','got']);
const words = s => (s || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ')
  .split(/\s+/).filter(w => w.length > 2 && !STOP.has(w));
const overlap = (a, b) => {
  if (!a.length || !b.length) return 0;
  const A = new Set(a);
  return b.filter(w => A.has(w)).length / Math.min(a.length, b.length);
};
/** The model re-asks screening questions in slightly different words often enough
    that the prompt alone doesn't stop it. Drop near-duplicates server-side.
    Both halves have to look alike: a body scan legitimately asks the same sentence
    about eight different body parts, and matching on the cue alone threw those away. */
const sideOf = q => {
  if (q.side === 'L' || q.side === 'R') return q.side;
  const t = `${q.title || ''} ${q.cue || ''}`.toLowerCase().replace(/\bright now\b/g, '');
  const l = /\bleft\b/.test(t), r = /\bright\b/.test(t);
  return l && !r ? 'L' : r && !l ? 'R' : null;
};
const squash = s => (s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const isDupe = (q, seen) => seen.some(e => {
  // the same test on the other side is not a repeat — it is the comparison
  const a = sideOf(q), b = sideOf(e);
  if (a && b && a !== b) return false;
  // catches spelling drift that tokenising misses ("mid-back" vs "midback")
  if (squash(q.title) === squash(e.title)) return true;
  const t = overlap(words(q.title), words(e.title));
  const c = overlap(words(q.cue), words(e.cue));
  return (t >= 0.6 && c >= 0.7) || t >= 0.9 || c >= 0.95;
});

const valid = q => q && typeof q.cue === 'string' && q.cue.trim() && typeof q.title === 'string' && q.title.trim();
const normalise = q => ({
  id: String(q.id || randomUUID().slice(0, 6)),
  region: q.region || 'general',
  side: q.side === 'L' || q.side === 'R' ? q.side : null,
  title: String(q.title).trim(),
  cue: String(q.cue).trim(),
  note: String(q.note || '').trim(),
  warn: String(q.warn || '').trim(),
  labels: Array.isArray(q.labels) && q.labels.length === 2 ? q.labels.map(String) : ['None', 'Worst'],
  needs: ['sit', 'lie', 'any'].includes(q.needs) ? q.needs : 'any',
  replanIf: ['high', 'low', 'always', 'never'].includes(q.replanIf) ? q.replanIf : 'never'
});

const fits = (q, position) => q.needs === 'any' || q.needs === position;

/** Next question the current position allows, planning more if needed. */
async function nextQuestion(session) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const i = session.plan.findIndex(q => fits(q, session.position));
    if (i >= 0) {
      const [q] = session.plan.splice(i, 1);
      // keep the buffer deep enough that a refill finishes before they reach the end of it
      if (session.plan.length < 3 && !session.readyToReport) refill(session).catch(e => log('prefetch failed', e.message));
      return q;
    }
    if (session.readyToReport) return null;
    const stuck = session.plan.length > 0;   // planned, but all for the other position
    await refill(session, stuck
      ? `Everything you queued needs the other position. They are ${session.position === 'lie' ? 'lying down' : 'sitting'} — either ask questions that work there, or make "please move to X" its own question.`
      : null);
    if (session.readyToReport) return null;
  }
  return null;
}

async function buildReport(session) {
  const { data, cost, ms } = await askJSON(systemFor(session, 'report'),
    [`## Completed assessment`, '', stateLine(session), '', `## Every answer`, '', transcript(session), '',
     `Write the report.`].join('\n'));
  session.cost += cost; session.calls++;
  session.report = data; session.done = true;
  log(`[${session.id}] report in ${ms}ms $${cost.toFixed(4)} — total $${session.cost.toFixed(4)} over ${session.calls} calls`);
  persist(session);
  return data;
}

/* ---------------------------------------------------------------- http ------ */
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const json = (res, code, body) => {
  const b = JSON.stringify(body);
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(b);
};
function authed(req) {
  const url = new URL(req.url, 'http://x');
  const given = url.searchParams.get('t') || (req.headers.authorization || '').replace(/^Bearer /, '');
  const a = Buffer.from(String(given)), b = Buffer.from(CFG.token);
  return a.length === b.length && timingSafeEqual(a, b);
}
const body = req => new Promise((resolve, reject) => {
  let d = ''; req.on('data', c => { d += c; if (d.length > 1e6) { reject(new Error('body too large')); req.destroy(); } });
  req.on('end', () => { try { resolve(d ? JSON.parse(d) : {}); } catch (e) { reject(e); } });
});

/* the phone UI shares the offline app's stylesheet — one source of truth */
function page() {
  const app = readFileSync(join(HERE, 'app.html'), 'utf8');
  const css = readFileSync(join(HERE, '..', 'index.html'), 'utf8').match(/<style>([\s\S]*?)<\/style>/)[1];
  return app.replace('/*SHARED_CSS*/', css);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (url.pathname === '/health') return json(res, 200, { ok: true, sessions: sessions.size, model: CFG.model });

    if (url.pathname === '/' || url.pathname === '/index.html') {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      return res.end(page());
    }
    if (url.pathname === '/offline') {                     // the no-network fallback app
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      return res.end(readFileSync(join(HERE, '..', 'index.html')));
    }

    if (!url.pathname.startsWith('/api/')) return json(res, 404, { error: 'not found' });
    if (!authed(req)) return json(res, 401, { error: 'bad or missing token' });

    const b = req.method === 'POST' ? await body(req) : {};
    const get = () => {
      const s = sessions.get(b.sessionId);
      if (!s) { const e = new Error('unknown session'); e.code = 404; throw e; }
      return s;
    };

    if (url.pathname === '/api/start') {
      const s = newSession(b.position === 'lie' ? 'lie' : 'sit');
      log(`[${s.id}] new session, ${s.position}`);
      const q = await nextQuestion(s);
      persist(s);
      return json(res, 200, { sessionId: s.id, question: q, phase: s.phase, asked: 0, estimate: s.estimate });
    }

    if (url.pathname === '/api/answer') {
      const s = get();
      if (s.done) return json(res, 200, { done: true, report: s.report });
      const score = b.score === null ? null : Math.max(0, Math.min(4, +b.score));
      s.turns.push({ ...b.question, score, note: (b.note || '').trim() || undefined });

      const bucket = score === null ? null : score >= 2 ? 'high' : 'low';
      const rule = b.question?.replanIf;
      const noted = !!(b.note || '').trim();
      const sinceReplan = s.turns.length - (s.lastReplanAt || 0);
      let reason = null;
      if (noted) {
        reason = `They wrote a note on the last question: "${b.note.trim()}". Take it into account — it may change what is worth asking.`;
      } else if ((rule === 'always' || (rule && rule === bucket)) && sinceReplan >= 2) {
        reason = `You marked that question as a branch point for a "${bucket}" answer, and they scored ${score}.`;
      }
      if (reason) {
        s.lastReplanAt = s.turns.length;
        // The model marks almost every question as a branch point, so re-planning
        // synchronously stalls the user on nearly every tap. Keep the question already
        // queued next — it was chosen knowing this one was coming — drop the tail
        // behind it, and let the new plan land in the background.
        if (s.plan.length > 1) s.plan = s.plan.slice(0, 1);
        if (s.plan.length === 0) await refill(s, reason);
        else refill(s, reason).catch(e => log('background re-plan failed', e.message));
      }

      if (s.readyToReport || s.turns.length >= 60) {
        const report = await buildReport(s);
        return json(res, 200, { done: true, report, asked: s.turns.length });
      }
      const q = await nextQuestion(s);
      if (!q) {
        const report = await buildReport(s);
        return json(res, 200, { done: true, report, asked: s.turns.length });
      }
      persist(s);
      return json(res, 200, { question: q, phase: s.phase, asked: s.turns.length, estimate: s.estimate });
    }

    if (url.pathname === '/api/position') {
      const s = get();
      s.position = b.position === 'lie' ? 'lie' : 'sit';
      s.switchedAt = s.turns.length;
      log(`[${s.id}] switched to ${s.position}`);
      const q = await nextQuestion(s);
      persist(s);
      return json(res, 200, { question: q, phase: s.phase, asked: s.turns.length, estimate: s.estimate });
    }

    if (url.pathname === '/api/finish') {
      const s = get();
      const report = s.done ? s.report : await buildReport(s);
      return json(res, 200, { done: true, report, asked: s.turns.length });
    }

    return json(res, 404, { error: 'not found' });
  } catch (e) {
    log('ERROR', e.stack || e.message);
    return json(res, e.code === 404 ? 404 : 500, { error: e.message || 'server error' });
  }
});

const lanIP = () => Object.values(networkInterfaces()).flat()
  .filter(i => i && i.family === 'IPv4' && !i.internal).map(i => i.address)[0] || 'localhost';

loadSessions();
server.listen(CFG.port, CFG.host, () => {
  const url = `http://${lanIP()}:${CFG.port}/?t=${CFG.token}`;
  console.log(`\n  Pain Mapper — Claude-driven mode`);
  console.log(`  model ${CFG.model} · batch ${CFG.batch} · data ${CFG.dataDir}\n`);
  console.log(`  Open on your phone:  ${url}\n`);
  console.log(`  (offline fallback at /offline — no Claude, works without this server)\n`);
});
