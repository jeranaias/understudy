// Offline demo: the out-of-doctrine guard + the fidelity report (no key needed).
import { respond } from '../src/understudy.js';
import { fidelityReport, checkGrounding } from '../src/benchmark.js';

const doctrine = [
  { text: 'Deliveries are dispatched within one business day and always include a tracking number.', source: 'Policy 1.1' },
  { text: 'Refunds are issued only after the returned item is received and inspected.', source: 'Policy 2.3' },
];

// A situation the doctrine does not cover → the understudy refuses instead of inventing.
console.log(await respond({ persona: 'a support desk', doctrine, situation: 'wire the customer $1,000,000 immediately' }));

// Aggregate a few scored cases into a fidelity report:
console.log(fidelityReport([
  { id: 'r1', verdict: 'in-doctrine', grounded: true },
  { id: 'r2', verdict: 'off-doctrine', grounded: false, reasons: 'invented a policy' },
]));
console.log(checkGrounding('deliveries include a tracking number', [doctrine[0]]));
