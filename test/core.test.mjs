import { test } from 'node:test';
import assert from 'node:assert/strict';
import { doctrineRetriever, respond } from '../src/understudy.js';
import { fidelityReport, checkGrounding, benchmark } from '../src/benchmark.js';

const doctrine = [
  { text: 'Supply convoys move only during daylight and always with an armed escort.', source: 'SOP 3.1' },
  { text: 'A checkpoint challenges every vehicle and verifies identification before allowing passage.', source: 'SOP 4.2' },
];

test('doctrineRetriever ranks the relevant passage', async () => {
  const r = doctrineRetriever(doctrine);
  const top = await r('how should a supply convoy move', 2);
  assert.equal(top[0].source, 'SOP 3.1');
});

test('respond returns out-of-doctrine (no model call) when nothing is retrieved', async () => {
  const r = await respond({ persona: 'a logistics cell', doctrine, situation: 'launch an orbital strike' });
  assert.equal(r.inDoctrine, false);
  assert.equal(r.note, 'out_of_doctrine');
  assert.deepEqual(r.citations, []);
});

test('respond validates inputs', async () => {
  await assert.rejects(() => respond({ doctrine, situation: 'x' }), TypeError); // no persona
});

test('checkGrounding detects overlap vs. drift', () => {
  const cites = [{ text: 'convoys move only during daylight with an armed escort' }];
  assert.equal(checkGrounding('the convoy moves during daylight with an escort', cites).grounded, true);
  assert.equal(checkGrounding('the unit conducts a naval blockade offshore', cites).grounded, false);
});

test('fidelityReport aggregates verdicts', () => {
  const rep = fidelityReport([
    { id: 'a', verdict: 'in-doctrine', grounded: true },
    { id: 'b', verdict: 'off-doctrine', grounded: false, reasons: 'invented a capability' },
    { id: 'c', verdict: 'partial', grounded: true },
  ]);
  assert.equal(rep.n, 3);
  assert.equal(rep.conforming, 1);
  assert.equal(rep.fidelity, 0.333);
  assert.equal(rep.byVerdict['off-doctrine'], 1);
  assert.equal(rep.groundedRate, 0.667);
  assert.equal(rep.failures.length, 2);
});

test('benchmark runs offline with an injected judge', async () => {
  // empty-retrieving doctrine forces the out-of-doctrine agent path (no model), injected judge scores it
  const fakeJudge = async () => ({ verdict: 'off-doctrine', reasons: 'no doctrine covered it' });
  const { report, runs } = await benchmark(
    [{ id: '1', situation: 'unrelated situation xyz', expect: 'do nothing' }],
    { persona: 'a cell', doctrine: [{ text: 'totally unrelated content about gardening', source: 'X' }], judge: fakeJudge },
  );
  assert.equal(report.n, 1);
  assert.equal(runs[0].verdict, 'off-doctrine');
});
