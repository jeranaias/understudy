import { test } from 'node:test';
import assert from 'node:assert/strict';
import { doctrineRetriever, respond, groundResponse } from '../src/understudy.js';
import { fidelityReport, checkGrounding, benchmark, scoreCase, normalizeVerdict, VERDICTS, UNKNOWN } from '../src/benchmark.js';
import { tokenize, stem } from '../src/grounding.js';

const doctrine = [
  { text: 'Supply convoys move only during daylight and always with an armed escort.', source: 'SOP 3.1' },
  { text: 'A checkpoint challenges every vehicle and verifies identification before allowing passage.', source: 'SOP 4.2' },
];

// ── doctrineRetriever ────────────────────────────────────────────────────────
test('doctrineRetriever ranks the relevant passage first', async () => {
  const r = doctrineRetriever(doctrine);
  const top = await r('how should a supply convoy move', 2);
  assert.equal(top[0].source, 'SOP 3.1');
  assert.ok(top[0].score > 0);
});

test('doctrineRetriever honors the k limit', async () => {
  const r = doctrineRetriever(doctrine);
  const top = await r('convoy checkpoint vehicle daylight escort identification', 1);
  assert.equal(top.length, 1);
});

test('doctrineRetriever returns [] when nothing matches', async () => {
  const r = doctrineRetriever(doctrine);
  assert.deepEqual(await r('quarterly marketing budget spreadsheet'), []);
});

test('doctrineRetriever accepts bare-string passages and the cite alias', async () => {
  const r = doctrineRetriever(['convoys move only during daylight', { text: 'checkpoint verifies identification', cite: 'SOP 4.2' }]);
  const top = await r('identification at the checkpoint', 5);
  assert.equal(top[0].source, 'SOP 4.2');
});

test('doctrineRetriever tolerates empty/undefined doctrine and rejects non-arrays', async () => {
  assert.deepEqual(await doctrineRetriever()('anything'), []);
  assert.deepEqual(await doctrineRetriever([])('anything'), []);
  assert.throws(() => doctrineRetriever('not an array'), TypeError);
});

// ── tokenize ─────────────────────────────────────────────────────────────────
test('tokenize drops punctuation, short words and stopwords', () => {
  assert.deepEqual(tokenize('The convoy, at night!'), ['convoy', 'night']);
  assert.deepEqual(tokenize(''), []);
  assert.deepEqual(tokenize(null), []);
});

// ── checkGrounding ───────────────────────────────────────────────────────────
test('checkGrounding detects overlap vs. drift', () => {
  const cites = [{ text: 'convoys move only during daylight with an armed escort' }];
  assert.equal(checkGrounding('the convoy moves during daylight with an escort', cites).grounded, true);
  assert.equal(checkGrounding('the unit conducts a naval blockade offshore', cites).grounded, false);
});

test('checkGrounding respects a custom threshold', () => {
  const cites = ['convoys move during daylight'];
  const g = checkGrounding('the convoy moves during daylight then departs elsewhere afterwards', cites);
  assert.ok(g.score > 0 && g.score < 1);
  assert.equal(checkGrounding('the convoy moves during daylight then departs elsewhere afterwards', cites, { threshold: 0.9 }).grounded, false);
  assert.equal(checkGrounding('the convoy moves during daylight then departs elsewhere afterwards', cites, { threshold: 0.1 }).grounded, true);
});

test('checkGrounding is empty/undefined safe', () => {
  assert.deepEqual(checkGrounding('', ['x']), { grounded: false, score: 0 });
  assert.deepEqual(checkGrounding('anything at all here', []), { grounded: false, score: 0 });
  assert.deepEqual(checkGrounding('anything at all here', undefined), { grounded: false, score: 0 });
});

test('checkGrounding accepts string and object citations alike', () => {
  assert.equal(checkGrounding('daylight escort convoy', ['daylight escort convoy']).score, 1);
  assert.equal(checkGrounding('daylight escort convoy', [{ text: 'daylight escort convoy' }]).score, 1);
});

// ── normalizeVerdict ─────────────────────────────────────────────────────────
test('normalizeVerdict maps the canonical and drifted forms', () => {
  assert.equal(normalizeVerdict('in-doctrine'), 'in-doctrine');
  assert.equal(normalizeVerdict('in doctrine'), 'in-doctrine');
  assert.equal(normalizeVerdict('IN_DOCTRINE'), 'in-doctrine');
  assert.equal(normalizeVerdict('conforms'), 'in-doctrine');
  assert.equal(normalizeVerdict('out of doctrine'), 'off-doctrine');
  assert.equal(normalizeVerdict('fail'), 'off-doctrine');
  assert.equal(normalizeVerdict('who knows'), 'unknown');
  assert.equal(normalizeVerdict(undefined), 'unknown');
  for (const v of VERDICTS) assert.equal(normalizeVerdict(v), v);
});

// ── fidelityReport ───────────────────────────────────────────────────────────
test('fidelityReport aggregates verdicts, grounding and failures', () => {
  const rep = fidelityReport([
    { id: 'a', verdict: 'in-doctrine', grounded: true, groundingScore: 0.8 },
    { id: 'b', verdict: 'off-doctrine', grounded: false, groundingScore: 0.1, reasons: 'invented a capability' },
    { id: 'c', verdict: 'partial', grounded: true, groundingScore: 0.6 },
  ]);
  assert.equal(rep.n, 3);
  assert.equal(rep.conforming, 1);
  assert.equal(rep.fidelity, 0.333);
  assert.equal(rep.byVerdict['off-doctrine'], 1);
  assert.equal(rep.groundedRate, 0.667);
  assert.equal(rep.meanGrounding, 0.5);
  assert.equal(rep.failures.length, 2);
  assert.equal(rep.failures[0].id, 'b');
});

test('fidelityReport handles the empty set', () => {
  const rep = fidelityReport([]);
  assert.equal(rep.n, 0);
  assert.equal(rep.fidelity, 0);
  assert.equal(rep.groundedRate, null);
  assert.equal(rep.meanGrounding, null);
  assert.deepEqual(rep.failures, []);
});

test('fidelityReport reports fidelity 1 when all conform', () => {
  const rep = fidelityReport([
    { id: 'a', verdict: 'in-doctrine' },
    { id: 'b', verdict: 'in-doctrine' },
  ]);
  assert.equal(rep.fidelity, 1);
  assert.equal(rep.failures.length, 0);
});

test('fidelityReport leaves groundedRate null when no grounded flags are present', () => {
  const rep = fidelityReport([{ id: 'a', verdict: 'partial' }]);
  assert.equal(rep.groundedRate, null);
  assert.equal(rep.meanGrounding, null);
});

test('fidelityReport normalizes unknown verdicts and throws on non-arrays', () => {
  const rep = fidelityReport([{ id: 'a', verdict: 'CONFORMS' }, { id: 'b', verdict: 'weird' }]);
  assert.equal(rep.conforming, 1);
  assert.equal(rep.byVerdict.unknown, 1); // 'weird' is flagged unknown, not silently bucketed as partial
  assert.equal(rep.byVerdict.partial, 0);
  assert.throws(() => fidelityReport('nope'), TypeError);
  assert.throws(() => fidelityReport(null), TypeError);
});

// ── groundResponse (pure finalizer + strict gate) ────────────────────────────
const passages = [
  { text: 'convoys move only during daylight and always with an armed escort', source: 'SOP 3.1' },
  { text: 'a checkpoint verifies identification before passage', source: 'SOP 4.2' },
];

test('groundResponse resolves cited passages and marks in-doctrine', () => {
  const r = groundResponse({ action: 'hold the convoy until daylight with an armed escort', rationale: 'per [1]', used: [1] }, passages);
  assert.equal(r.inDoctrine, true);
  assert.equal(r.citations.length, 1);
  assert.equal(r.citations[0].source, 'SOP 3.1');
  assert.ok(r.grounding.score > 0);
});

test('groundResponse falls back to the top passages when the model cites none', () => {
  const r = groundResponse({ action: 'verify identification at the checkpoint', rationale: 'x' }, passages);
  assert.equal(r.citations.length, 2);
});

test('groundResponse passes the model out-of-doctrine flag through as a refusal', () => {
  const r = groundResponse({ outOfDoctrine: true, note: 'out_of_doctrine' }, passages);
  assert.equal(r.inDoctrine, false);
  assert.equal(r.note, 'out_of_doctrine');
  assert.deepEqual(r.citations, []);
});

test('groundResponse surfaces a model error', () => {
  const r = groundResponse({ error: 'HTTP 500' }, passages);
  assert.equal(r.inDoctrine, false);
  assert.equal(r.note, 'HTTP 500');
});

test('strict mode downgrades a weakly-grounded answer to a refusal', () => {
  const drift = { action: 'conduct a naval blockade offshore near the harbor entrance', rationale: 'tactical judgment', used: [1] };
  assert.equal(groundResponse(drift, passages).inDoctrine, true); // permissive by default
  const strict = groundResponse(drift, passages, { strict: true });
  assert.equal(strict.inDoctrine, false);
  assert.equal(strict.note, 'weak_grounding');
});

test('strict mode keeps a well-grounded answer', () => {
  const good = { action: 'hold the convoy until daylight and assign an armed escort', rationale: 'per [1]', used: [1] };
  assert.equal(groundResponse(good, passages, { strict: true }).inDoctrine, true);
});

// ── respond ──────────────────────────────────────────────────────────────────
test('respond returns out-of-doctrine (no model call) when nothing is retrieved', async () => {
  const r = await respond({ persona: 'a logistics cell', doctrine, situation: 'launch an orbital strike' });
  assert.equal(r.inDoctrine, false);
  assert.equal(r.note, 'out_of_doctrine');
  assert.deepEqual(r.citations, []);
  assert.deepEqual(r.grounding, { grounded: false, score: 0 });
});

test('respond validates inputs', async () => {
  await assert.rejects(() => respond({ doctrine, situation: 'x' }), TypeError); // no persona
  await assert.rejects(() => respond({ persona: 'p' }), TypeError); // no situation
  await assert.rejects(() => respond({ persona: 'p', situation: 'x' }), TypeError); // no doctrine/retriever
  await assert.rejects(() => respond({ persona: 'p', situation: 'x', retriever: 'nope' }), TypeError);
  await assert.rejects(() => respond(), TypeError);
});

test('respond accepts a custom retriever and refuses when it returns nothing', async () => {
  const emptyRetriever = async () => [];
  const r = await respond({ persona: 'a cell', retriever: emptyRetriever, situation: 'anything' });
  assert.equal(r.note, 'out_of_doctrine');
});

// ── benchmark ────────────────────────────────────────────────────────────────
test('benchmark validates inputs', async () => {
  await assert.rejects(() => benchmark([], { persona: 'p' }), TypeError);
  await assert.rejects(() => benchmark('nope', { persona: 'p' }), TypeError);
  await assert.rejects(() => benchmark([{ situation: 's', expect: 'e' }], {}), TypeError); // no persona
  await assert.rejects(() => benchmark([{ expect: 'e' }], { persona: 'p' }), TypeError); // no situation
  await assert.rejects(() => benchmark([{ situation: 's', expect: 'e' }], { persona: 'p', judge: 5 }), TypeError);
});

test('benchmark runs offline with an injected judge and aggregates multiple cases', async () => {
  // empty-retrieving doctrine forces the out-of-doctrine agent path (no model); injected judge scores it
  const doc = [{ text: 'gardening notes about roses and tulips', source: 'X' }];
  const calls = [];
  const fakeJudge = async ({ situation, response }) => {
    calls.push({ situation, response });
    return { verdict: situation.includes('ok') ? 'in-doctrine' : 'off-doctrine', reasons: 'injected' };
  };
  const { report, runs } = await benchmark(
    [
      { id: '1', situation: 'unrelated situation xyz', expect: 'do nothing' },
      { id: '2', situation: 'ok situation abc', expect: 'do nothing' },
    ],
    { persona: 'a cell', doctrine: doc, judge: fakeJudge },
  );
  assert.equal(report.n, 2);
  assert.equal(report.conforming, 1);
  assert.equal(report.fidelity, 0.5);
  assert.equal(runs[0].verdict, 'off-doctrine');
  assert.equal(runs[1].verdict, 'in-doctrine');
  assert.equal(typeof runs[0].groundingScore, 'number');
  assert.equal(calls.length, 2);
});

test('benchmark normalizes a drifted judge verdict', async () => {
  const { runs } = await benchmark(
    [{ id: '1', situation: 'unrelated xyz', expect: 'nothing' }],
    { persona: 'a cell', doctrine: [{ text: 'gardening notes about roses' }], judge: async () => ({ verdict: 'CONFORMS' }) },
  );
  assert.equal(runs[0].verdict, 'in-doctrine');
});

// ── scoreCase (no key path) ──────────────────────────────────────────────────
test('scoreCase returns an errored (not off-doctrine) verdict when no model key is set', async () => {
  const prevA = process.env.UNDERSTUDY_API_KEY;
  const prevB = process.env.OPENROUTER_API_KEY;
  delete process.env.UNDERSTUDY_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const r = await scoreCase({ situation: 's', response: 'r', expect: 'e', doctrine });
    assert.equal(r.verdict, 'unknown'); // a judge model error is errored, NOT a fabricated off-doctrine
    assert.equal(r.errored, true);
    assert.equal(r.conforms, false);
    assert.match(r.reasons, /UNDERSTUDY_API_KEY/);
  } finally {
    if (prevA !== undefined) process.env.UNDERSTUDY_API_KEY = prevA;
    if (prevB !== undefined) process.env.OPENROUTER_API_KEY = prevB;
  }
});

// ── HARDENING: normalizeVerdict whitespace + expanded synonyms ─────────────────
test('normalizeVerdict trims BEFORE dashing so leading/trailing space keeps the verdict', () => {
  assert.equal(normalizeVerdict(' in-doctrine'), 'in-doctrine'); // was mis-bucketed to partial
  assert.equal(normalizeVerdict('in-doctrine '), 'in-doctrine');
  assert.equal(normalizeVerdict('  off doctrine  '), 'off-doctrine');
  assert.equal(normalizeVerdict('-in-doctrine-'), 'in-doctrine');
});

test('normalizeVerdict recognizes off/in synonyms instead of swallowing them as partial', () => {
  for (const off of ['non-conforming', 'noncompliant', 'violation', 'violates', 'breach', 'rejected', 'no'])
    assert.equal(normalizeVerdict(off), 'off-doctrine', off);
  for (const inn of ['compliant', 'yes', 'conforms', 'passed', 'accepted'])
    assert.equal(normalizeVerdict(inn), 'in-doctrine', inn);
  assert.equal(normalizeVerdict('partially'), 'partial');
  assert.equal(normalizeVerdict('banana'), UNKNOWN); // genuinely unknown is flagged, not partial
});

// ── HARDENING: error vs. refusal disambiguation + benchmark denominator ────────
test('groundResponse marks a model error errored:true with the contract action, refusal errored:false', () => {
  const err = groundResponse({ error: 'HTTP 500' }, passages);
  assert.equal(err.errored, true);
  assert.equal(err.inDoctrine, false);
  assert.equal(err.action, 'No action taken.'); // documented contract, was '' before
  assert.equal(err.note, 'HTTP 500');

  const ref = groundResponse({ outOfDoctrine: true, note: 'out_of_doctrine' }, passages);
  assert.equal(ref.errored, false); // a deliberate refusal is not an error
  assert.equal(ref.action, 'No action taken.');
});

test('fidelityReport EXCLUDES errored runs from the fidelity denominator', () => {
  const rep = fidelityReport([
    { id: 'a', verdict: 'in-doctrine' },
    { id: 'b', verdict: 'off-doctrine', reasons: 'drift' },
    { id: 'e', errored: true, verdict: UNKNOWN, reasons: 'HTTP 500' },
  ]);
  assert.equal(rep.n, 3);
  assert.equal(rep.errored, 1);
  assert.equal(rep.scored, 2);
  assert.equal(rep.fidelity, 0.5); // 1 of 2 scored, NOT 1 of 3
  assert.equal(rep.failures.length, 1); // the errored run is not a doctrine failure
  assert.equal(rep.byVerdict.unknown, 0); // errored run excluded from the verdict tally
});

test('benchmark counts an agent model error as errored, never as an off-doctrine failure', async () => {
  const erroringChat = async () => ({ error: 'HTTP 503' });
  const { report, runs } = await benchmark(
    [{ id: '1', situation: 'a supply convoy at night', expect: 'hold until daylight' }],
    { persona: 'a cell', doctrine, chat: erroringChat, judge: async () => ({ verdict: 'off-doctrine' }) },
  );
  assert.equal(runs[0].errored, true);
  assert.equal(report.errored, 1);
  assert.equal(report.scored, 0);
  assert.equal(report.fidelity, 0);
  assert.equal(report.failures.length, 0); // errored, not scored off-doctrine
});

// ── HARDENING: unicode tokenizer + stem-tolerant grounding ─────────────────────
test('tokenize keeps non-English (accented Latin, Cyrillic, Arabic) tokens', () => {
  assert.deepEqual(tokenize('El convoy avanza al mediodía con escolta'), ['convoy', 'avanza', 'mediodía', 'con', 'escolta']);
  assert.deepEqual(tokenize('Конвой движется днём'), ['конвой', 'движется', 'днём']);
  assert.deepEqual(tokenize('القافلة تتحرك نهارا'), ['القافلة', 'تتحرك', 'نهارا']);
});

test('a non-English SOP retrieves and grounds instead of stripping to noise', async () => {
  const ar = [{ text: 'القافلة تتحرك نهارا فقط مع حراسة مسلحة', source: 'SOP-AR' }];
  const top = await doctrineRetriever(ar)('متى تتحرك القافلة نهارا', 5);
  assert.equal(top[0].source, 'SOP-AR');
  assert.ok(top[0].score > 0);
  assert.ok(checkGrounding('القافلة تتحرك نهارا', ar).score > 0);
});

test('grounding is stem-tolerant across inflection', () => {
  assert.equal(stem('convoys'), 'convoy');
  assert.equal(stem('moves'), 'move');
  assert.equal(stem('escorting'), 'escort');
  assert.equal(stem('passes'), 'pass');
  // plural/singular drift still counts as the same grounded evidence
  assert.equal(checkGrounding('the convoys and escorts', [{ text: 'convoy escort' }]).score, 1);
});

// ── HARDENING: empty doctrine refuses (does not throw) ─────────────────────────
test('respond refuses (not throws) on an empty doctrine array — nothing is permitted', async () => {
  const r = await respond({ persona: 'a cell', doctrine: [], situation: 'do anything' });
  assert.equal(r.inDoctrine, false);
  assert.equal(r.errored, false);
  assert.equal(r.note, 'out_of_doctrine');
  // still throws when neither doctrine nor retriever is provided at all
  await assert.rejects(() => respond({ persona: 'p', situation: 'x' }), TypeError);
});

// ── HARDENING: k validation ────────────────────────────────────────────────────
test('respond errors (contract action) on an invalid k instead of silently retrieving 0', async () => {
  for (const bad of [NaN, 'five', -1, Infinity]) {
    const r = await respond({ persona: 'a cell', doctrine, situation: 'a supply convoy at night', k: bad });
    assert.equal(r.errored, true, String(bad));
    assert.equal(r.note, 'invalid_k', String(bad));
    assert.equal(r.action, 'No action taken.', String(bad));
  }
});

// ── HARDENING: strict gate enforcement ─────────────────────────────────────────
test('strict mode refuses an empty action even if the leftover text looks grounded', () => {
  const empty = { action: '   ', rationale: 'convoys move only during daylight with an armed escort', used: [1] };
  assert.equal(groundResponse(empty, passages).inDoctrine, true); // permissive default keeps it
  const strict = groundResponse(empty, passages, { strict: true });
  assert.equal(strict.inDoctrine, false);
  assert.equal(strict.note, 'weak_grounding');
});

// ── HARDENING: per-case isolation ──────────────────────────────────────────────
test('benchmark isolates a throwing case and still scores the rest', async () => {
  const flakyJudge = async ({ situation }) => {
    if (situation.includes('boom')) throw new Error('judge exploded');
    return { verdict: 'in-doctrine' };
  };
  const doc = [{ text: 'supply convoys move only during daylight with an armed escort', source: 'SOP 3.1' }];
  const agentChat = async () => ({ action: 'hold the convoy until daylight with an armed escort', rationale: 'per [1]', used: [1] });
  const { report, runs } = await benchmark(
    [
      { id: 'ok1', situation: 'a supply convoy at night', expect: 'hold until daylight' },
      { id: 'bad', situation: 'boom a supply convoy', expect: 'hold until daylight' },
      { id: 'ok2', situation: 'an armed escort for the convoy', expect: 'assign escort' },
    ],
    { persona: 'a cell', doctrine: doc, chat: agentChat, judge: flakyJudge },
  );
  assert.equal(runs.length, 3); // the throw did not abort the run
  assert.equal(report.n, 3);
  assert.equal(report.errored, 1);
  assert.equal(report.scored, 2);
  assert.equal(report.conforming, 2);
  assert.equal(report.fidelity, 1); // 2/2 scored conform; the errored case is excluded
  assert.equal(runs.find((r) => r.id === 'bad').errored, true);
});

// ── HARDENING: injectable chat (respond model path is testable, no network) ────
test('respond runs the full model path with an injected chat and grounds the decision', async () => {
  const cannedChat = async (system) => {
    assert.match(system, /Act STRICTLY/); // it really reached the model path
    return { action: 'Hold the convoy until daylight and assign an armed escort.', rationale: 'per [1]', used: [1], outOfDoctrine: false };
  };
  const r = await respond({ persona: 'a logistics cell', doctrine, situation: 'a supply convoy at night', chat: cannedChat });
  assert.equal(r.inDoctrine, true);
  assert.equal(r.errored, false);
  assert.ok(r.grounding.score > 0);
  assert.equal(r.citations[0].source, 'SOP 3.1');
});

test('scoreCase judges only the cited retrieved subset it is handed, via an injected chat', async () => {
  let seen = '';
  const judgeChat = async (system, user) => {
    seen = user;
    return { verdict: 'in-doctrine', reasons: 'ok' };
  };
  const r = await scoreCase({
    situation: 's', response: 'hold the convoy', rationale: 'per [1]', expect: 'hold',
    citations: [{ text: 'convoys move only during daylight' }], chat: judgeChat,
  });
  assert.equal(r.verdict, 'in-doctrine');
  assert.equal(r.errored, false);
  assert.match(seen, /convoys move only during daylight/);
  assert.match(seen, /Agent rationale: per \[1\]/); // the rationale is fed to the judge
});
