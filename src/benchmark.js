// Verification & validation for a doctrine-bound agent. An agent that "acts per doctrine" is only
// trustworthy if you can prove it — so Understudy ships the benchmark harness alongside the agent.
// Give it cases (a situation + the doctrinally-correct behavior you expect); it runs the agent, has a
// judge score each response for conformance, checks grounding, and aggregates a fidelity report.
import { respond } from './understudy.js';
import { checkGrounding } from './grounding.js';

const ENDPOINT = process.env.UNDERSTUDY_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';
// The judge defaults to a DIFFERENT model than the agent (understudy.js → UNDERSTUDY_MODEL): scoring an
// agent with its own weights leaks the agent's blind spots into the grade. Override with UNDERSTUDY_JUDGE_MODEL.
const JUDGE_MODEL = process.env.UNDERSTUDY_JUDGE_MODEL || 'openai/gpt-4o-mini';

/** The conformance verdicts, best to worst. `unknown` is a distinct off-scale bucket for a verdict the
 * judge did not express intelligibly — it is never silently folded into `partial`. */
export const VERDICTS = ['in-doctrine', 'partial', 'off-doctrine'];
export const UNKNOWN = 'unknown';

// Explicit synonym sets. Enforce, don't soften: a clear off-doctrine signal ("violation", "noncompliant")
// must not be swallowed as "partial", and only a genuinely unrecognizable verdict falls through to `unknown`.
const IN = new Set(['in-doctrine', 'conforms', 'conform', 'conformant', 'conforming', 'compliant', 'compliance', 'pass', 'passes', 'passed', 'yes', 'ok', 'accept', 'accepted', 'correct']);
const OFF = new Set(['off-doctrine', 'out-of-doctrine', 'off', 'fail', 'fails', 'failed', 'violates', 'violate', 'violation', 'non-conforming', 'nonconforming', 'noncompliant', 'non-compliant', 'noncompliance', 'breach', 'reject', 'rejected', 'no', 'incorrect', 'invents']);
const PARTIAL = new Set(['partial', 'partially', 'mixed', 'somewhat', 'borderline', 'incomplete', 'unclear']);

/**
 * Normalize a judge's raw verdict onto the {@link VERDICTS} scale. PURE. Order matters: lowercase →
 * trim → collapse whitespace/underscores to dashes → strip stray edge dashes, so `' in-doctrine'`
 * normalizes to `in-doctrine` (not `partial`). Recognizes an explicit synonym set for each verdict;
 * a verdict it cannot place returns `unknown` — flagged, never quietly bucketed as `partial`.
 * @param {string} v
 * @returns {'in-doctrine'|'partial'|'off-doctrine'|'unknown'}
 */
export function normalizeVerdict(v) {
  const s = String(v || '')
    .toLowerCase()
    .trim()
    .replace(/[\s_]+/g, '-')
    .replace(/^-+|-+$/g, '');
  if (VERDICTS.includes(s)) return s;
  if (IN.has(s)) return 'in-doctrine';
  if (OFF.has(s)) return 'off-doctrine';
  if (PARTIAL.has(s)) return 'partial';
  return UNKNOWN;
}

/** The default judge model call. Injectable via {@link scoreCase}'s `chat` for offline/deterministic
 * scoring. Uses {@link JUDGE_MODEL} (independent of the agent) at `temperature: 0` for reproducibility. */
async function callJudge(system, user, timeoutMs = 45000) {
  const KEY = process.env.UNDERSTUDY_API_KEY || process.env.OPENROUTER_API_KEY;
  if (!KEY) return { error: 'set UNDERSTUDY_API_KEY (or OPENROUTER_API_KEY)' };
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      signal: AbortSignal.timeout(timeoutMs),
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: JUDGE_MODEL,
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
 * Runs flagged `errored: true` (a model/network failure, not a doctrine outcome) are EXCLUDED from the
 * fidelity denominator — they are counted separately, never scored as off-doctrine.
 * @param {{id?:string, verdict:string, errored?:boolean, grounded?:boolean, groundingScore?:number, reasons?:string}[]} results
 * @returns {{ n:number, scored:number, errored:number, conforming:number, fidelity:number,
 *   groundedRate:number|null, meanGrounding:number|null, byVerdict:Record<string,number>, failures:object[] }}
 */
export function fidelityReport(results) {
  if (!Array.isArray(results)) throw new TypeError('fidelityReport(results): results must be an array');
  const n = results.length;
  const buckets = [...VERDICTS, UNKNOWN];
  const byVerdict = Object.fromEntries(buckets.map((v) => [v, 0]));
  let conforming = 0;
  let errored = 0;
  let groundedN = 0;
  let groundedTrue = 0;
  let scoreN = 0;
  let scoreSum = 0;
  const failures = [];
  for (const r of results) {
    if (r.errored) {
      errored++; // excluded from the denominator, byVerdict, grounding and failures
      continue;
    }
    const verdict = buckets.includes(r.verdict) ? r.verdict : normalizeVerdict(r.verdict);
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
  const scored = n - errored;
  return {
    n,
    scored,
    errored,
    conforming,
    fidelity: scored ? Math.round((conforming / scored) * 1000) / 1000 : 0,
    groundedRate: groundedN ? Math.round((groundedTrue / groundedN) * 1000) / 1000 : null,
    meanGrounding: scoreN ? Math.round((scoreSum / scoreN) * 1000) / 1000 : null,
    byVerdict,
    failures,
  };
}

/**
 * Judge one response against the doctrinally-correct behavior expected for its situation. Uses an LLM
 * examiner (a model independent of the agent; see {@link JUDGE_MODEL}). The judge sees only the RETRIEVED
 * subset the agent actually cited plus the agent's rationale — not the entire doctrine corpus — so it
 * grades what the agent grounded on, and a large doctrine can't dilute the examination. A model/network
 * failure returns `errored: true` (verdict `unknown`), never a fabricated `off-doctrine`.
 * @param {{situation:string, response:string, rationale?:string, expect:string,
 *   citations?:any[], doctrine?:any[],
 *   chat?:(system:string,user:string)=>Promise<object>}} args
 * @returns {Promise<{verdict:string, conforms:boolean, reasons:string, errored:boolean}>}
 */
export async function scoreCase({ situation, response, rationale, expect, citations, doctrine, chat = callJudge }) {
  const passages = citations || doctrine || []; // the retrieved/cited subset, not the whole corpus
  const ctx = passages.map((d, i) => `[${i + 1}] ${typeof d === 'string' ? d : d.text}`).join('\n\n');
  const sys = `You are a strict doctrine examiner. Given the cited doctrine passages, a situation, the response an agent produced (with its rationale), and the doctrinally-correct behavior expected, judge conformance. "in-doctrine" = matches the expected behavior and stays within the cited doctrine; "partial" = partially correct or missing key elements; "off-doctrine" = contradicts the doctrine or invents beyond it. Output JSON only: {"verdict":"in-doctrine|partial|off-doctrine","conforms":true,"reasons":"..."}`;
  const r = await chat(sys, `Cited doctrine:\n${ctx}\n\nSituation: ${situation}\nExpected doctrinal behavior: ${expect}\nAgent response: ${response}\nAgent rationale: ${rationale || ''}\n\nJudge it.`);
  if (r.error) return { verdict: UNKNOWN, conforms: false, reasons: r.error, errored: true };
  const verdict = normalizeVerdict(r.verdict);
  return { verdict, conforms: verdict === 'in-doctrine', reasons: r.reasons || '', errored: false };
}

/**
 * Run a full V&V benchmark: for each case, have the agent respond, judge conformance, check grounding,
 * and aggregate. Bring a custom `judge` for offline/deterministic scoring.
 * @param {{id?:string, situation:string, expect:string}[]} cases
 * @param {{ persona:string, doctrine?:any[], retriever?:Function, k?:number, strict?:boolean,
 *   groundingThreshold?:number, chat?:Function,
 *   judge?:(c:{situation:string,response:string,rationale?:string,expect:string,citations?:any[]})=>Promise<{verdict:string,reasons?:string,errored?:boolean}> }} opts
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
    // Per-case isolation: one throwing case is recorded as errored and excluded from the denominator,
    // it must never abort the whole run.
    try {
      const r = await respond({
        persona: opts.persona,
        doctrine: opts.doctrine,
        retriever: opts.retriever,
        situation: c.situation,
        k: opts.k,
        strict: opts.strict,
        groundingThreshold: opts.groundingThreshold,
        chat: opts.chat,
      });
      // The agent itself errored (model/network) — mark it errored, don't judge an empty response.
      if (r.errored) {
        runs.push({ id: c.id, response: r, verdict: UNKNOWN, conforms: false, reasons: r.note, errored: true, grounded: false, groundingScore: 0 });
        results.push({ id: c.id, verdict: UNKNOWN, errored: true, grounded: false, groundingScore: 0, reasons: r.note });
        continue;
      }
      // Feed the judge the agent's rationale and the retrieved subset it cited — not the whole corpus.
      const scored = await judge({ situation: c.situation, response: r.action, rationale: r.rationale, expect: c.expect, citations: r.citations, doctrine: opts.doctrine });
      const verdict = normalizeVerdict(scored.verdict);
      const errored = !!scored.errored;
      const g = r.grounding || checkGrounding(`${r.action} ${r.rationale}`, r.citations);
      runs.push({ id: c.id, response: r, ...scored, verdict, errored, grounded: g.grounded, groundingScore: g.score });
      results.push({ id: c.id, verdict, errored, grounded: g.grounded, groundingScore: g.score, reasons: scored.reasons });
    } catch (e) {
      const reason = e && e.message ? e.message : String(e);
      runs.push({ id: c.id, response: null, verdict: UNKNOWN, conforms: false, reasons: reason, errored: true, grounded: false, groundingScore: 0 });
      results.push({ id: c.id, verdict: UNKNOWN, errored: true, grounded: false, groundingScore: 0, reasons: reason });
    }
  }
  return { report: fidelityReport(results), runs };
}

export { checkGrounding };
