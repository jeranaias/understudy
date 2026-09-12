# 🎭 Understudy

[![CI](https://github.com/jeranaias/understudy/actions/workflows/ci.yml/badge.svg)](https://github.com/jeranaias/understudy/actions/workflows/ci.yml) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**An agent that stands in for a scarce subject-matter expert — acting strictly within a body of doctrine, with a benchmark to prove it.**

When the human expert isn't in the room, you still need someone to play the part — an adversary cell in
a wargame, a specialist reviewer, any role governed by a body of doctrine or standard operating
procedures. Understudy plays it, but on a leash: it acts **only** in accordance with the doctrine you
give it, cites the passages it's applying, and **refuses to invent** a capability or procedure the
doctrine doesn't grant.

And because "it acts per doctrine" is worthless unless you can *prove* it, Understudy ships a **fidelity
benchmark** alongside the agent.

```js
import { respond } from 'understudy';

await respond({
  persona: 'an adversary logistics cell',
  doctrine: [{ text: 'Convoys move only in daylight, always with an armed escort.', source: 'SOP 3.1' }],
  situation: 'A resupply is needed at the forward position tonight.',
});
// → { action: 'Hold the convoy until first light and assign an armed escort…',
//     rationale: '… [1]', inDoctrine: true, citations: [{ source: 'SOP 3.1' }] }
```

Ask it for something the doctrine doesn't cover and it won't make it up:

```js
// → { action: 'No action taken.', inDoctrine: false, note: 'out_of_doctrine', citations: [] }
```

## Prove the fidelity (V&V)

```js
import { benchmark } from 'understudy';

const { report } = await benchmark(cases, { persona, doctrine });
// cases: [{ situation, expect: 'the doctrinally-correct behavior' }]
// → report: { n, conforming, fidelity: 0.86, groundedRate, byVerdict, failures }
```

`fidelityReport` (the aggregator) and `checkGrounding` are pure — no key, fully unit-tested.

## Install & test

```bash
npm install understudy
export UNDERSTUDY_API_KEY=...   # any OpenAI-compatible key (OpenRouter by default)
node example/demo.mjs           # the offline guards + report
npm test
```

## License

Apache-2.0.
