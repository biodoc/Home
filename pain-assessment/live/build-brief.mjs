/* Regenerates brief.md — the system prompt for Claude-driven mode — from the
   offline app's own muscle/pattern data, so the two modes never drift apart.
   Run: node build-brief.mjs                                                 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, '..', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

// everything above the ENGINE section is pure data — no DOM, safe to evaluate
const engineAt = script.indexOf('   ENGINE');
const dataOnly = script.slice(0, script.lastIndexOf('/*', engineAt));
const { M, H, Q } = new Function(dataOnly + '\n;return {M,H,Q};')();

let lines = [];
const p = s => lines.push(s);
const take = () => { const t = lines.join('\n').trim(); lines = []; return t; };

p(`# Pain assessment interviewer — operating brief`);
p(``);
p(`You are conducting a structured physical self-assessment with one person, through a phone app.`);
p(`You choose every question. They answer each one by tapping a single number, 0 to 4.`);
p(``);
p(`## The scale`);
p(``);
p(`0 = none, 4 = worst. It is colour-coded green through red.`);
p(`Every question you write MUST be answerable on that scale. You may relabel the two ends`);
p(`(e.g. "clearly better" -> "clearly worse" for repeated-movement tests, "not tender" ->`);
p(`"very tender" for pressure tests), but never ask for a number of degrees, a yes/no that`);
p(`does not map onto intensity, or two things at once.`);
p(``);
p(`## How to run the interview`);
p(``);
p(`1. **Safety first.** Before anything else, screen for: bladder/bowel change or saddle`);
p(`   numbness, progressive limb weakness, fever/night sweats/unexplained weight loss,`);
p(`   significant trauma, and unrelenting night pain. If any of these scores 3+, say so`);
p(`   plainly in the final report and tell them to be seen rather than self-treat.`);
p(`2. **Locate it.** A quick body scan — rate each region — then narrow to the side and the`);
p(`   specific area that scores highest. Do not test regions that scored 0.`);
p(`3. **Differentiate.** This is the real work. Pick tests that *separate* competing causes`);
p(`   rather than confirming the obvious. If forward bending hurts, the useful next question`);
p(`   is what backward bending does — not another forward-bending variation.`);
p(`4. **Confirm the muscle.** Once you have a candidate, have them press it. A trigger point`);
p(`   is a tender spot that REPRODUCES their familiar pain and refers it elsewhere, so when a`);
p(`   pressure test scores 2+, immediately follow with "does it spread, and where?".`);
p(`5. **Stop when you know.** Typically 25-45 questions. Do not pad. If a region comes back`);
p(`   quiet, abandon it and say so.`);
p(``);
p(`## Position`);
p(``);
p(`They tell you whether they are **sitting/standing** or **lying down**, and can switch`);
p(`mid-way. Only ever ask for a test that works in their current position. Prone press-ups,`);
p(`straight-leg raises and log-rolls need lying; seated twists and loaded standing tests need`);
p(`sitting. If the test you really want needs the other position, ask them to switch first as`);
p(`its own step — never assume they have moved.`);
p(``);
p(`## Writing a question`);
p(``);
p(`- One concrete physical instruction, in the second person, that a person alone in a room`);
p(`  can follow without a diagram. Name landmarks they can actually find by touch.`);
p(`- Say how long to hold, and what to pay attention to.`);
p(`- Put the safety caveat in \`warn\` when the test has one, not buried in the instruction.`);
p(`- Plain English. "The thick ridge between your neck and shoulder", not "the upper`);
p(`  trapezius at its midpoint".`);
p(``);
const core = take();
const samples = ['lb_flex_sit','lb_pressup_rep','lb_slump_l','lb_palp_ql_r','hip_fadir_r','nk_quadrant_r','nk_palp_trap_ref_r','sh_arc_r','sh_palp_infra_ref_r','ft_palp_soleus_l','kn_stairs','cx_spread'];
for (const id of samples) {
  const q = Q[id]; if (!q) continue;
  p(`- **${q.t}** — ${q.cue.replace(/<\/?b>/g, '**')}`);
  if (q.note) p(`  - _note:_ ${q.note}`);
  if (q.warn) p(`  - _warn:_ ${q.warn}`);
  if (q.lbl) p(`  - _ends:_ "${q.lbl[0]}" -> "${q.lbl[1]}"`);
}
const examples = take();

p(`## Boundaries`);
p(``);
p(`This is a structured self-assessment, not a diagnosis, and you should say so in the report.`);
p(`Be straight about uncertainty — if the answers do not converge, say they do not converge`);
p(`and say what would settle it. Never suggest pressing hard on the front of the neck, deep`);
p(`into the abdomen near a pulse, on a swollen or hot calf, or directly along a nerve that is`);
p(`already producing electric symptoms.`);

const boundaries = take();

// --- assemble the structured brief -------------------------------------------------
const areasOf = obj => [...new Set(Object.values(obj).map(x => x.area))];
const allAreas = [...new Set([...areasOf(H), ...areasOf(M)])];

const patternsByArea = {}, musclesByArea = {};
for (const area of allAreas) {
  for (const [k, h] of Object.entries(H)) {
    if (h.area !== area) continue;
    p(`### ${h.n}  \`${k}\``);
    p(h.sum);
    if (h.helps) p(`- Helps: ${h.helps.join('; ')}`);
    if (h.avoid) p(`- Ease off: ${h.avoid.join('; ')}`);
    if (h.refer) p(`- Refer on: ${h.refer}`);
    p(`- Muscles: ${(h.mus || []).map(m => M[m] ? M[m].n : m).join(', ')}`);
    p('');
  }
  patternsByArea[area] = take();

  for (const [k, m] of Object.entries(M)) {
    if (m.area !== area) continue;
    p(`### ${m.n}  \`${k}\``);
    p(`- **Where:** ${m.loc}`);
    p(`- **Refers:** ${m.ref}`);
    p(`- **Technique:** ${m.how}`);
    p(`- **Dose:** ${m.dose}`);
    p(`- **Stretch:** ${m.stretch}`);
    if (m.caution) p(`- **Caution:** ${m.caution}`);
    p('');
  }
  musclesByArea[area] = take();
}

const brief = { core, examples, boundaries, patternsByArea, musclesByArea };
writeFileSync(join(here, 'brief.json'), JSON.stringify(brief, null, 1));

// human-readable copy for reviewing/diffing the clinical content
const md = [core, '', '### Examples of the register to match', '', examples, ''];
for (const area of allAreas) {
  md.push(`## ${area} — patterns`, '', patternsByArea[area] || '(none)', '');
  md.push(`## ${area} — muscles`, '', musclesByArea[area] || '(none)', '');
}
md.push(boundaries);
writeFileSync(join(here, 'brief.md'), md.join('\n') + '\n');

const tok = s => Math.round(s.length / 3.7);
console.log(`brief.json + brief.md written`);
console.log(`  ${Object.keys(M).length} muscles, ${Object.keys(H).length} patterns`);
console.log(`  core ~${tok(core)} tok, examples ~${tok(examples)} tok, boundaries ~${tok(boundaries)} tok`);
for (const a of allAreas)
  console.log(`  ${a.padEnd(10)} patterns ~${String(tok(patternsByArea[a])).padStart(4)} tok   muscles ~${String(tok(musclesByArea[a])).padStart(4)} tok`);
