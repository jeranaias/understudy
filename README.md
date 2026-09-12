# 🎭 Understudy

[![CI](https://github.com/jeranaias/understudy/actions/workflows/ci.yml/badge.svg)](https://github.com/jeranaias/understudy/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**An agent that stands in for a scarce subject-matter expert — acting strictly within a body of doctrine, with a benchmark to prove it.**

When the expert isn't in the room, you still need someone to play the part — an adversary cell in a
wargame, a policy reviewer, a first-line responder, any role governed by a body of doctrine or standard
operating procedures. Understudy plays it, but **on a leash**: it acts *only* in accordance with the
doctrine you give it, cites the passages it's applying, and **refuses to invent** a capability or
procedure the doctrine doesn't grant.

The value isn't "an LLM role-plays." It's "an LLM role-plays **on a leash, and you can prove it stayed
on the leash**." Three things make that real: **grounding** (every decision cites the passages it rests
on), **refusal** (out of doctrine → it declines, it doesn't improvise), and a **fidelity benchmark** that
measures conformance so the claim is a number, not a vibe.

```js
import { respond } from 'understudy';

await respond({
  persona: 'an adversary logistics cell',
  doctrine: [{ text: 'Convoys move only in daylight, always with an armed escort.', source: 'SOP 3.1' }],
  situation: 'A resupply is needed at the forward position tonight.',
});
// → { action: 'Hold the convoy until first light and assign an armed escort…',
//     rationale: '… [1]', inDoctrine: true,
//     citations: [{ source: 'SOP 3.1', text: 'Convoys move only in daylight…' }],
//     grounding: { grounded: true, score: 0.42 } }
```

Ask it for something the doctrine doesn't cover and it won't make it up:

```js
await respond({ persona: 'an adversary logistics cell', doctrine, situation: 'Call in an orbital strike.' });
// → { action: 'No action taken.', rationale: 'The doctrine does not address this situation.',
//     inDoctrine: false, note: 'out_of_doctrine', citations: [], grounding: { grounded: false, score: 0 } }
```

Tighten the leash one more notch with **strict mode** — even an on-topic answer is downgraded to a
refusal if it can't be sufficiently grounded in the passages it cited:

```js
await respond({ persona, doctrine, situation, strict: true, groundingThreshold: 0.3 });
// weakly-grounded → { action: 'No action taken.', inDoctrine: false, note: 'weak_grounding', … }
```

## Prove the fidelity (V&V)

An agent that "acts per doctrine" is worthless unless you can *prove* it — so Understudy ships the
verification harness alongside the agent. Give it cases (a situation + the doctrinally-correct behavior
you expect); it runs the agent, has a judge score each response for conformance, checks grounding, and
aggregates a fidelity report.

```js
import { benchmark } from 'understudy';

const cases = [
  { id: 'night-resupply', situation: 'A resupply is needed at the forward position tonight.',
    expect: 'Hold the convoy until daylight and assign an armed escort.' },
];

const { report, runs } = await benchmark(cases, { persona: 'an adversary logistics cell', doctrine });
// report → { n: 1, conforming: 1, fidelity: 1, groundedRate: 1, meanGrounding: 0.44,
//            byVerdict: { 'in-doctrine': 1, partial: 0, 'off-doctrine': 0 }, failures: [] }
// runs   → [{ id, response, verdict, conforms, reasons, grounded, groundingScore }]
```

The aggregator and the grounding proxy are **pure** — no key, fully unit-tested — so you can grade and
brief from them anywhere:

```js
import { fidelityReport, checkGrounding } from 'understudy';

checkGrounding('the convoy moves in daylight with an escort',
  [{ text: 'convoys move only in daylight with an armed escort' }]);
// → { grounded: true, score: 0.5 }

fidelityReport([
  { verdict: 'in-doctrine', grounded: true, groundingScore: 0.7 },
  { verdict: 'off-doctrine', grounded: false, groundingScore: 0.1, reasons: 'invented a capability' },
]);
// → { n: 2, conforming: 1, fidelity: 0.5, groundedRate: 0.5, meanGrounding: 0.4, byVerdict: {…}, failures: [{…}] }
```

### Why fidelity, not vibes

"It felt in-character" doesn't survive a review. A fidelity number does. The benchmark turns *does the
stand-in stay on doctrine?* into a repeatable measurement — conformance rate, a per-verdict breakdown,
and a traceable list of every failure — so you can regression-test a persona the same way you'd test
code, and defend the result instead of asserting it.

## Data shapes

**Doctrine passage** — a string, or `{ text, source? }` (`cite` is accepted as an alias for `source`).

**Case** (`benchmark` input):

```ts
{ id?: string, situation: string, expect: string }  // expect = the doctrinally-correct behavior
```

**Response** (`respond` output):

```ts
{
  action: string,               // what the entity does / decides ('No action taken.' on a refusal)
  rationale: string,            // why, per the doctrine, with [n] cites
  inDoctrine: boolean,          // false on any refusal
  note: string,                 // '' | 'out_of_doctrine' | 'weak_grounding' | an error string
  citations: { source: string, text: string }[],
  grounding: { grounded: boolean, score: number }   // overlap of the answer with its citations, 0..1
}
```

## API

| Export | Kind | Purpose |
| --- | --- | --- |
| `respond({ persona, doctrine \| retriever, situation, k?, strict?, groundingThreshold? })` | async | The stand-in acts on a situation, grounded in doctrine; refuses out of doctrine. |
| `doctrineRetriever(doctrine)` | sync → fn | Zero-dependency keyword retriever; `(query, k=5) => passages`. |
| `groundResponse(raw, passages, { strict?, groundingThreshold? })` | **pure** | Turn a raw model decision into the final grounded response (citations + strict gate). Compose your own pipeline. |
| `benchmark(cases, { persona, doctrine \| retriever, k?, strict?, groundingThreshold?, judge? })` | async | Run the agent over cases, judge conformance, aggregate a fidelity report. |
| `scoreCase({ situation, response, expect, doctrine? })` | async | The LLM judge for one case (inject your own into `benchmark` for offline scoring). |
| `fidelityReport(results)` | **pure** | Aggregate scored cases into the fidelity summary. |
| `checkGrounding(text, citations, { threshold? })` | **pure** | Groundedness proxy: fraction of the answer's words found in its citations. |
| `normalizeVerdict(v)` | **pure** | Normalize a judge verdict onto `in-doctrine \| partial \| off-doctrine`. |

Bring your own retriever (e.g. a vector store) via `retriever: (query, k) => Promise<passages>` instead
of `doctrine`. Inject a deterministic `judge: (c) => ({ verdict, reasons? })` to run the benchmark with
no model at all.

## CLI

```bash
understudy respond --doctrine sop.txt --persona "a support desk" --situation "…" [--strict] [--k 5] [--json]
understudy bench   cases.json --doctrine sop.txt --persona "a support desk" [--strict] [--json]
```

A doctrine file is plain text with passages separated by blank lines; a cases file is a JSON array of
`{ situation, expect }`.

## Configuration

Model calls go to any OpenAI-compatible chat-completions endpoint.

| Env var | Default | Purpose |
| --- | --- | --- |
| `UNDERSTUDY_API_KEY` | — | API key (`OPENROUTER_API_KEY` is also accepted). |
| `UNDERSTUDY_ENDPOINT` | `https://openrouter.ai/api/v1/chat/completions` | Chat-completions URL. |
| `UNDERSTUDY_MODEL` | `google/gemini-3-flash-preview` | Model id. |

## Install & test

```bash
npm install understudy
export UNDERSTUDY_API_KEY=...   # any OpenAI-compatible key (OpenRouter by default)
node example/demo.mjs           # the offline guards + strict mode + report (no key needed)
npm test
```

## License

Apache-2.0.
