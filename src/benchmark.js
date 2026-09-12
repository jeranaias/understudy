// Verification & validation for a doctrine-bound agent. An agent that "acts per doctrine" is only
// trustworthy if you can prove it — so Understudy ships the benchmark harness alongside the agent.
// Give it cases (a situation + the doctrinally-correct behavior you expect); it runs the agent, has a
// judge score each response for conformance, and aggregates a fidelity report.
import { respond } from './understudy.js';

const ENDPOINT = process.env.UNDERSTUDY_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = process.env.UNDERSTUDY_MODEL || 'google/gemini-3-flash-preview';
const VERDICTS = ['in-doctrine', 'partial', 'off-doctrine'];

async function chat(system, user, timeoutMs = 45000) {
  const KEY = process.env.UNDERSTUDY_API_KEY || process.env.OPENROUTER_API_KEY;
  if (!KEY) return { error: 'set UNDERSTUDY_API_KEY' };
  try {
    const res = await fetch(ENDPOINT, { method: 'POST', signal: AbortSignal.timeout(timeoutMs),
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, temperature: 0, messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }) });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    const t = (await res.json()).choices?.[0]?.message?.content ?? '';
    const m = t.match(/\{[\s\S]*\}/); if (m) { try { return JSON.parse(m[0]); } catch {} }
    return { error: 'no-json' };
  } catch (e) { return { error: e?.name === 'AbortError' ? 'timeout' : String(e) }; }
}

const STOP = new Set('a an the of to and or in on for with by is are be as at from that this it its into their'.split(' '));
const toks = (s) => new Set(String(s || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter((w) => w.length > 2 && !STOP.has(w)));

/**
 * Groundedness proxy — how much of a response's content actually overlaps the doctrine it cited.
 * Pure; a cheap guard against a response that cites doctrine but wanders beyond it.
 * @param {string} responseText
 * @param {(string|{text:string})[]} citations
 * @param {{ threshold?: number }} [opts]  overlap fraction to count as grounded (default 0.3)
 * @returns {{ grounded: boolean, score: number }}
 */
export function checkGrounding(responseText, citations, opts = {}) {
  const threshold = opts.threshold ?? 0.3;
  const r = toks(responseText);
  if (!r.size) return { grounded: false, score: 0 };
  const d = toks((citations || []).map((c) => (typeof c === 'string' ? c : c.text)).join(' '));
  let hit = 0; r.forEach((w) => { if (d.has(w)) hit++; });
  const score = Math.round((hit / r.size) * 100) / 100;
  return { grounded: score >= threshold, score };
}

/**
 * Aggregate scored cases into a fidelity report. PURE — the trustworthy summary you brief from.
 * @param {{id?:string, verdict:string, grounded?:boolean}[]} results
 * @returns {{ n:number, conforming:number, fidelity:number, groundedRate:number|null,
 *   byVerdict:object, failures:object[] }}
 */
export function fidelityReport(results) {
  if (!Array.isArray(results)) throw new TypeError('fidelityReport(results): results must be an array');
  const n = results.length;
  const byVerdict = Object.fromEntries(VERDICTS.map((v) => [v, 0]));
  let conforming = 0, groundedN = 0, groundedTrue = 0;
  const failures = [];
  for (const r of results) {
    if (VERDICTS.includes(r.verdict)) byVerdict[r.verdict]++;
    if (r.verdict === 'in-doctrine') conforming++;
    else failures.push({ id: r.id, verdict: r.verdict, reasons: r.reasons });
    if (typeof r.grounded === 'boolean') { groundedN++; if (r.grounded) groundedTrue++; }
  }
  return {
    n,
    conforming,
    fidelity: n ? Math.round((conforming / n) * 1000) / 1000 : 0,
    groundedRate: groundedN ? Math.round((groundedTrue / groundedN) * 1000) / 1000 : null,
    byVerdict,
    failures,
  };
}

/**
 * Judge one response against the doctrinally-correct behavior expected for its situation.
 * @returns {Promise<{verdict:string, conforms:boolean, reasons:string}>}
 */
export async function scoreCase({ situation, response, expect, doctrine }) {
  const ctx = (doctrine || []).map((d, i) => `[${i + 1}] ${typeof d === 'string' ? d : d.text}`).join('\n\n');
  const sys = `You are a strict doctrine examiner. Given the doctrine, a situation, the response an agent produced, and the doctrinally-correct behavior expected, judge conformance. "in-doctrine" = matches the expected behavior and stays within the doctrine; "partial" = partially correct or missing key elements; "off-doctrine" = contradicts the doctrine or invents beyond it. Output JSON only: {"verdict":"in-doctrine|partial|off-doctrine","conforms":true,"reasons":"..."}`;
  const r = await chat(sys, `Doctrine:\n${ctx}\n\nSituation: ${situation}\nExpected doctrinal behavior: ${expect}\nAgent response: ${response}\n\nJudge it.`);
  if (r.error) return { verdict: 'off-doctrine', conforms: false, reasons: r.error };
  const verdict = VERDICTS.includes(r.verdict) ? r.verdict : 'partial';
  return { verdict, conforms: verdict === 'in-doctrine', reasons: r.reasons || '' };
}

/**
 * Run a full V&V benchmark: for each case, have the agent respond, judge conformance, check
 * grounding, and aggregate. Bring a custom judge for offline/deterministic scoring.
 * @param {{id?:string, situation:string, expect:string}[]} cases
 * @param {{ persona:string, doctrine?:any[], retriever?:Function, k?:number,
 *   judge?:(c:object)=>Promise<{verdict:string,reasons?:string}> }} opts
 * @returns {Promise<{report:object, runs:object[]}>}
 */
export async function benchmark(cases, opts = {}) {
  if (!Array.isArray(cases) || !cases.length) throw new TypeError('benchmark(cases): non-empty array required');
  if (!opts.persona) throw new TypeError('benchmark: opts.persona required');
  const judge = opts.judge || scoreCase;
  const runs = [];
  const results = [];
  for (const c of cases) {
    const r = await respond({ persona: opts.persona, doctrine: opts.doctrine, retriever: opts.retriever, situation: c.situation, k: opts.k });
    const scored = await judge({ situation: c.situation, response: r.action, expect: c.expect, doctrine: opts.doctrine });
    const g = checkGrounding(r.action + ' ' + r.rationale, r.citations);
    runs.push({ id: c.id, response: r, ...scored, grounded: g.grounded, groundingScore: g.score });
    results.push({ id: c.id, verdict: scored.verdict, grounded: g.grounded, reasons: scored.reasons });
  }
  return { report: fidelityReport(results), runs };
}
