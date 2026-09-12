// Offline demo — every part here runs with no API key. It exercises the leash-and-proof thesis:
// the out-of-doctrine refusal, the pure finalizer (groundResponse) with a strict grounding gate,
// the fidelity report, and the grounding proxy.
import { respond, groundResponse } from '../src/understudy.js';
import { fidelityReport, checkGrounding } from '../src/benchmark.js';

const doctrine = [
  { text: 'Deliveries are dispatched within one business day and always include a tracking number.', source: 'Policy 1.1' },
  { text: 'Refunds are issued only after the returned item is received and inspected.', source: 'Policy 2.3' },
];

// 1. A situation the doctrine does not cover → the understudy refuses instead of inventing.
console.log('— out-of-doctrine refusal —');
console.log(await respond({ persona: 'a support desk', doctrine, situation: 'wire the customer $1,000,000 immediately' }));

// 2. groundResponse is the pure finalizer — feed it a model decision (here a canned one) and it
//    resolves citations and measures grounding without any network call.
console.log('\n— grounded decision (pure) —');
const passages = doctrine.map((d) => ({ text: d.text, source: d.source }));
console.log(groundResponse({ action: 'Issue the refund once the returned item is received and inspected.', rationale: 'per [2]', used: [2] }, passages));

// 3. strict mode downgrades a decision that drifts off the cited doctrine to a refusal.
console.log('\n— strict mode catches drift —');
console.log(groundResponse({ action: 'Overnight a replacement drone by private courier.', rationale: 'goodwill', used: [1] }, passages, { strict: true }));

// 4. Aggregate a few scored cases into a fidelity report.
console.log('\n— fidelity report —');
console.log(fidelityReport([
  { id: 'r1', verdict: 'in-doctrine', grounded: true, groundingScore: 0.7 },
  { id: 'r2', verdict: 'off-doctrine', grounded: false, groundingScore: 0.1, reasons: 'invented a policy' },
]));

// 5. The grounding proxy on its own.
console.log('\n— grounding proxy —');
console.log(checkGrounding('deliveries include a tracking number', [doctrine[0]]));
