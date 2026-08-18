# Pain Mapper

A self-contained pain assessment you run on your phone. It walks you through movement
and pressure tests one at a time, you answer each with a single tap on a **0–4** scale
(green 0 → red 4), and at the end it tells you the likely pattern **and exactly which
muscles to massage, where they are, and how to work them.**

No API, no account, no network. One HTML file, ~120 KB, zero external requests.

## Run it

**On a phone (the intended way)**

1. Copy `index.html` onto the phone — AirDrop, email it to yourself, or put it in
   iCloud/Drive/Dropbox and open it from there.
2. Open it in Safari or Chrome. That's it.

Because the file makes no network requests at all, it keeps working in airplane mode.
On iOS, *Share → Add to Home Screen* gives it an app icon and hides the browser chrome.

**From a computer**

```bash
open pain-assessment/index.html        # macOS
xdg-open pain-assessment/index.html    # Linux
```

Or serve the folder if you'd rather use a URL: `npx http-server pain-assessment`

Your answers and saved reports live in that browser's localStorage only. Nothing leaves
the device.

## How it works

**One input, always.** Every prompt takes a 0–4 tap. Colours are fixed: 0 green, 4 red.
Some questions relabel the ends — "clearly better → clearly worse" for repeated-movement
tests, "not tender → very tender" for pressure tests — but it's always the same five
buttons. Keyboard `0`–`4` works too, plus `←` back, `→` skip, `n` for a note.

**Sitting or lying.** Pick at the start, change any time with **Adjust**. Tests are
filtered to what's actually possible in your position — prone press-ups and straight-leg
raises only appear when you're lying down, seated twists only when you're up. Switch
mid-assessment and the remaining questions are re-planned around you; if the question on
screen no longer works in the new position, it steps past it. When you reach the end with
useful tests still locked behind the other position, it offers to switch and continue.

**It adapts as you go.**

- Score high on a movement and the follow-ups that *separate* the possible causes get
  inserted immediately — rate the slump test 3 and you're asked to lift your chin, because
  if that eases the leg pain it's nerve tension rather than a tight hamstring.
- Score low across a whole region and the rest of that region is dropped.
- Repeated-movement tests ("do ten press-ups, now compare") test directional preference,
  which is what actually separates a flexion-sensitive back from an extension-sensitive one.
- **Your notes change the questions.** Tap *Add note*, write "tingling shooting down my
  leg", and the nerve-tension tests are added on the spot. About twenty keyword rules do
  this — numbness, headache, night pain, groin, heel, grip, stairs, jaw, cough, injury,
  weakness, and so on.

**Trigger points are tested, not guessed.** Each region includes pressure tests on the
muscles that commonly refer pain there. If pressing reproduces your pain, a follow-up asks
whether it *spread* somewhere else — a tender spot plus referral is what actually defines a
trigger point. Those muscles are marked **confirmed tender** / **referred your pain** in the
results and sorted to the top of the massage plan.

## What you get at the end

- A body map coloured by region score, and a score table.
- Up to five ranked patterns with a confidence bar and what each means — e.g. flexion- vs
  extension-sensitive low back, SI joint, gluteal/lateral hip, deep buttock, nerve-involved
  leg pain, rotator cuff, cervicogenic headache, plantar/heel. Each comes with what helps and
  what to ease off. Patterns are only reported for regions you actually reported and tested.
- **Where to massage** — the main event. For every target muscle: where it is in findable
  landmarks, where it refers pain to and why that matches your answers, how to press it,
  how long and how often, the stretch that follows, and any caution specific to that muscle.
- A 10-minute routine sequencing the top three muscles: press, stretch, then move.
- Your notes, a saved history to compare against, and a plain-text report you can save or
  copy for a physio appointment.

Referral patterns follow the standard myofascial trigger point maps (Travell & Simons).

## Safety

Five screening questions run first. Cauda equina signs (bladder/bowel changes, saddle
numbness), progressive weakness, fever/night sweats/weight loss, significant trauma, and
unrelenting night pain all raise a banner on the results telling you to get seen rather than
self-treat. Individual tests carry their own cautions where they matter — the psoas near the
aorta, the scalenes over the brachial plexus, the armpit for subscapularis, calf work and
blood clots, the QL over the kidney.

This is a structured self-assessment, not a diagnosis. It can be wrong. Severe, spreading,
post-injury, or non-settling pain needs a real clinician.

## Editing the content

Everything lives in `index.html` in four plain data structures near the top of the `<script>`:

| What | Where | Holds |
|---|---|---|
| `M` | muscle library | location, referral, technique, dose, stretch, caution |
| `H` | patterns | description, helps, ease off, and which muscles it points at |
| `Q` | test bank | prompt text, allowed positions, evidence weights, branching rules |
| `KW` | keyword rules | which notes add which questions |

A test contributes evidence as `ev` (weight applied when you score high) and `evL` (applied
when you score low — a pain-free extension test is evidence *for* a flexion-sensitive back).
Confidence is the weighted fraction of the maximum a pattern could have scored, shrunk by
how few questions fed it, then gated to regions you actually assessed. Add a test with
`def()`, or `defSide()` to generate left and right variants from one definition.
