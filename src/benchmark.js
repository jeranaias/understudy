// Verification & validation for a doctrine-bound agent. An agent that "acts per doctrine" is only
// trustworthy if you can prove it — so Understudy ships the benchmark harness alongside the agent.
// Give it cases (a situation + the doctrinally-correct behavior you expect); it runs the agent, has a
// judge score each response for conformance, checks grounding, and aggregates a fidelity report.
import { respond } from './understudy.js';
import { checkGrounding } from './grounding.js';

const ENDPOINT = process.env.UNDERSTUDY_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = process.env.UNDERSTUDY_MODEL || 'google/gemini-3-flash-preview';

/** The conformance verdicts, best to worst. */
export const VERDICTS = ['in-doctrine', 'partial', 'off-doctrine'];

/**
 * Normalize a judge's raw verdict onto the {@link VERDICTS} scale. PURE. Accepts light spelling drift
 * ("in doctrine", "in_doctrine", "conforms", "pass") and falls back to `partial` for anything unknown,
 * so a chatty judge never crashes the aggregate.
 * @param {string} v
 * @returns {'in-doctrine'|'partial'|'off-doctrine'}
 */
export function normalizeVerdict(v) {
  const s = String(v || '').toLowerCase().replace(/[\s_]+/g, '-').trim();
  if (VERDICTS.includes(s)) return s;
  if (s === 'conforms' || s === 'conform' || s === 'pass' || s === 'in-doctrine-') return 'in-doctrine';
  if (s === 'off' || s === 'fail' || s === 'out-of-doctrine' || s === 'violates') return 'off-doctrine';
  return 'partial';
}

async function chat(system, user, timeoutMs = 45000) {
  const KEY = process.env.UNDERSTUDY_API_KEY || process.env.OPENROUTER_API_KEY;
  if (!KEY) return { error: 'set UNDERSTUDY_API_KEY (or OPENROUTER_API_KEY)' };
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      signal: AbortSignal.timeout(timeoutMs),
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: user },
        ],
      }),
    });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    const t = (await res.json()).choices?.[0]?.message?.content ?? '';
    const m = t.match(/\{[\s\S]*\}/);
    if (m) {
      try {
        return JSON.parse(m[0]);
      } catch {}
    }
    return { error: 'no-json' };
  } catch (e) {
    return { error: e?.name === 'AbortError' ? 'timeout' : String(e) };
  }
}

/**
 * Aggregate scored cases into a fidelity report. PURE — the trustworthy summary you brief from. Counts
 * conformance, breaks it down by verdict, rolls up grounding, and lists the failures so a red mark is
 * always traceable back to a case.
 * @param {{id?:string, verdict:string, grounded?:boolean, groundingScore?:number, reasons?:string}[]} results
 * @returns {{ n:number, conforming:number, fidelity:number, groundedRate:number|null,
 *   meanGrounding:number|null, byVerdict:Record<string,number>, failures:object[] }}
 */
export function fidelityReport(results) {
  if (!Array.isArray(results)) throw new TypeError('fidelityReport(results): results must be an array');
  const n = results.length;
  const byVerdict = Object.fromEntries(VERDICTS.map((v) => [v, 0]));
  let conforming = 0;
  let groundedN = 0;
  let groundedTrue = 0;
  let scoreN = 0;
  let scoreSum = 0;
  const failures = [];
  for (const r of results) {
    const verdict = VERDICTS.includes(r.verdict) ? r.verdict : normalizeVerdict(r.verdict);
    byVerdict[verdict]++;
    if (verdict === 'in-doctrine') conforming++;
    else failures.push({ id: r.id, verdict, reasons: r.reasons });
    if (typeof r.grounded === 'boolean') {
      groundedN++;
      if (r.grounded) groundedTrue++;
    }
    if (typeof r.groundingScore === 'number' && Number.isFinite(r.groundingScore)) {
      scoreN++;
      scoreSum += r.groundingScore;
    }
  }
  return {
    n,
    conforming,
    fidelity: n ? Math.round((conforming / n) * 1000) / 1000 : 0,
    groundedRate: groundedN ? Math.round((groundedTrue / groundedN) * 1000) / 1000 : null,
    meanGrounding: scoreN ? Math.round((scoreSum / scoreN) * 1000) / 1000 : null,
    byVerdict,
    failures,
  };
}

/**
 * Judge one response against the doctrinally-correct behavior expected for its situation. Uses an LLM
 * examiner; the verdict is normalized onto the {@link VERDICTS} scale. Requires a model key — inject a
 * custom judge into {@link benchmark} for offline/deterministic scoring.
 * @param {{situation:string, response:string, expect:string, doctrine?:any[]}} args
 * @returns {Promise<{verdict:string, conforms:boolean, reasons:string}>}
 */
export async function scoreCase({ situation, response, expect, doctrine }) {
  const ctx = (doctrine || []).map((d, i) => `[${i + 1}] ${typeof d === 'string' ? d : d.text}`).join('\n\n');
  const sys = `You are a strict doctrine examiner. Given the doctrine, a situation, the response an agent produced, and the doctrinally-correct behavior expected, judge conformance. "in-doctrine" = matches the expected behavior and stays within the doctrine; "partial" = partially correct or missing key elements; "off-doctrine" = contradicts the doctrine or invents beyond it. Output JSON only: {"verdict":"in-doctrine|partial|off-doctrine","conforms":true,"reasons":"..."}`;
  const r = await chat(sys, `Doctrine:\n${ctx}\n\nSituation: ${situation}\nExpected doctrinal behavior: ${expect}\nAgent response: ${response}\n\nJudge it.`);
  if (r.error) return { verdict: 'off-doctrine', conforms: false, reasons: r.error };
  const verdict = normalizeVerdict(r.verdict);
  return { verdict, conforms: verdict === 'in-doctrine', reasons: r.reasons || '' };
}

/**
 * Run a full V&V benchmark: for each case, have the agent respond, judge conformance, check grounding,
 * and aggregate. Bring a custom `judge` for offline/deterministic scoring.
 * @param {{id?:string, situation:string, expect:string}[]} cases
 * @param {{ persona:string, doctrine?:any[], retriever?:Function, k?:number, strict?:boolean,
 *   groundingThreshold?:number,
 *   judge?:(c:{situation:string,response:string,expect:string,doctrine?:any[]})=>Promise<{verdict:string,reasons?:string}> }} opts
 * @returns {Promise<{report:object, runs:object[]}>}
 */
export async function benchmark(cases, opts = {}) {
  if (!Array.isArray(cases) || !cases.length) throw new TypeError('benchmark(cases): non-empty array required');
  if (!opts || !opts.persona || typeof opts.persona !== 'string') throw new TypeError('benchmark: opts.persona (string) required');
  if (opts.judge != null && typeof opts.judge !== 'function') throw new TypeError('benchmark: opts.judge must be a function');
  for (const [i, c] of cases.entries()) {
    if (!c || typeof c.situation !== 'string' || !c.situation) throw new TypeError(`benchmark: cases[${i}].situation (string) required`);
  }
  const judge = opts.judge || scoreCase;
  const runs = [];
  const results = [];
  for (const c of cases) {
    const r = await respond({
      persona: opts.persona,
      doctrine: opts.doctrine,
      retriever: opts.retriever,
      situation: c.situation,
      k: opts.k,
      strict: opts.strict,
      groundingThreshold: opts.groundingThreshold,
    });
    const scored = await judge({ situation: c.situation, response: r.action, expect: c.expect, doctrine: opts.doctrine });
    const verdict = normalizeVerdict(scored.verdict);
    const g = r.grounding || checkGrounding(`${r.action} ${r.rationale}`, r.citations);
    runs.push({ id: c.id, response: r, ...scored, verdict, grounded: g.grounded, groundingScore: g.score });
    results.push({ id: c.id, verdict, grounded: g.grounded, groundingScore: g.score, reasons: scored.reasons });
  }
  return { report: fidelityReport(results), runs };
}

export { checkGrounding };
