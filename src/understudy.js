// Understudy — an agent that stands in for a scarce subject-matter expert by acting strictly in
// accordance with a body of doctrine / standard operating procedures. It answers or decides AS the
// entity you describe, grounded only in the provided doctrine, and refuses to invent a capability or
// procedure the doctrine doesn't grant. Model calls go to any OpenAI-compatible chat endpoint.
import { tokenize, tokenSet, checkGrounding } from './grounding.js';

const ENDPOINT = process.env.UNDERSTUDY_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = process.env.UNDERSTUDY_MODEL || 'google/gemini-3-flash-preview';

const asDoc = (p) =>
  typeof p === 'string' ? { text: p, source: '' } : { text: (p && p.text) || '', source: (p && (p.source || p.cite)) || '' };

/**
 * Zero-dependency keyword retriever over the doctrine passages. Scores each passage by the fraction of
 * the query's content words it contains, drops non-matches, and returns the top `k` — highest first.
 * @param {(string|{text:string, source?:string, cite?:string})[]} doctrine  the passages
 * @returns {(query:string, k?:number)=>Promise<{text:string, source:string, score:number}[]>}
 */
export function doctrineRetriever(doctrine) {
  if (doctrine != null && !Array.isArray(doctrine)) {
    throw new TypeError('doctrineRetriever(doctrine): doctrine must be an array of passages');
  }
  const docs = (doctrine || []).map(asDoc);
  return async (query, k = 5) => {
    const q = tokenSet(query);
    return docs
      .map((d) => {
        const t = tokenSet(d.text);
        let s = 0;
        q.forEach((w) => {
          if (t.has(w)) s++;
        });
        return { ...d, score: q.size ? Math.round((s / q.size) * 100) / 100 : 0 };
      })
      .filter((d) => d.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(0, k | 0));
  };
}

/**
 * The default model call — an OpenAI-compatible chat completion. Injectable: pass your own
 * `chat(system, user) => Promise<parsedJSON|{error}>` into {@link respond} to test the model path
 * (or swap providers) with no network. `temperature: 0` keeps the stand-in reproducible.
 */
async function callModel(system, user, timeoutMs = 45000) {
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
    try {
      return JSON.parse(t);
    } catch {}
    return { error: 'no-json' };
  } catch (e) {
    return { error: e?.name === 'AbortError' ? 'timeout' : String(e) };
  }
}

/** The refusal every out-of-doctrine path returns. A refusal is a *deliberate* decline, not an error
 * (`errored: false`). @param {string} note @returns {object} */
function refusal(note, rationale) {
  return {
    action: 'No action taken.',
    rationale: rationale || 'The doctrine does not address this situation.',
    inDoctrine: false,
    errored: false,
    note,
    citations: [],
    grounding: { grounded: false, score: 0 },
  };
}

/** A model/network/input error — shape-compatible with a refusal (`action: 'No action taken.'` per the
 * documented contract) but flagged `errored: true` so the benchmark can EXCLUDE it from the fidelity
 * denominator instead of miscounting it as a doctrine failure. @param {string} note @returns {object} */
function errorResponse(note, rationale) {
  return {
    action: 'No action taken.',
    rationale: rationale || '',
    inDoctrine: false,
    errored: true,
    note,
    citations: [],
    grounding: { grounded: false, score: 0 },
  };
}

/**
 * Turn a raw model decision into the final grounded response. PURE — no model call — so the whole
 * finalization contract (citation resolution, the out-of-doctrine flag, and the strict grounding gate)
 * is unit-testable without a key. Bring your own model call and hand its parsed JSON here to compose a
 * custom pipeline.
 * @param {{action?:string, rationale?:string, used?:number[], outOfDoctrine?:boolean, note?:string}} raw
 *   the model's parsed decision
 * @param {{text:string, source?:string}[]} passages  the retrieved passages the model saw, in order
 * @param {{ strict?:boolean, groundingThreshold?:number }} [opts]
 *   `strict` refuses a would-be in-doctrine answer unless it clears the whole gate: the model did NOT
 *   itself flag `outOfDoctrine`, the answer is non-empty and actually cites a passage, AND its grounding
 *   score is at or above `groundingThreshold` (default 0.3). The 0.3 floor is deliberately low — the
 *   proxy is lexical, so a well-grounded paraphrase can still score modestly; 0.3 catches wholesale
 *   drift into un-cited vocabulary without punishing honest rewording. Raise it for a tighter leash.
 * @returns {{action:string, rationale:string, inDoctrine:boolean, errored:boolean, note:string,
 *   citations:{source:string, text:string}[], grounding:{grounded:boolean, score:number}}}
 */
export function groundResponse(raw, passages, opts = {}) {
  const top = Array.isArray(passages) ? passages : [];
  const out = raw || {};
  if (out.error) return errorResponse(out.error);
  if (out.outOfDoctrine) return refusal(out.note || 'out_of_doctrine', out.rationale);

  const used = (Array.isArray(out.used) ? out.used : []).map((n) => top[n - 1]).filter(Boolean);
  const cite = used.length ? used : top.slice(0, 2);
  const citations = cite.map((d) => ({ source: d.source || 'doctrine', text: d.text }));
  const action = (out.action || '').trim();
  const rationale = out.rationale || '';
  const grounding = checkGrounding(`${action} ${rationale}`, citations, { threshold: opts.groundingThreshold });

  // Strict gate: enforce, don't caveat. A strict in-doctrine answer must be non-empty, cite doctrine,
  // and be lexically grounded — otherwise it is downgraded to a refusal.
  if (opts.strict && (!grounding.grounded || !action || !citations.length)) {
    return {
      ...refusal('weak_grounding', 'The response could not be sufficiently grounded in the cited doctrine.'),
      grounding,
    };
  }
  return { action: out.action || '', rationale, inDoctrine: true, errored: false, note: out.note || '', citations, grounding };
}

/**
 * Have the understudy respond to a situation, acting as `persona`, grounded ONLY in the doctrine. It
 * retrieves the most relevant passages, decides as the entity within them, cites what it applied, and
 * refuses (`inDoctrine: false`, `note: 'out_of_doctrine'`) rather than invent a capability the doctrine
 * doesn't grant.
 * @param {object} args
 * @param {string} args.persona   who the agent is playing (e.g. "an adversary logistics cell")
 * @param {(string|{text:string, source?:string})[]} [args.doctrine]  the doctrine/SOP passages
 * @param {(q:string, k:number)=>Promise<{text:string, source?:string, score?:number}[]>} [args.retriever]
 *   or bring your own retriever (e.g. a vector store); one of `doctrine` or `retriever` is required
 * @param {string} args.situation  the scenario/prompt to act on
 * @param {number} [args.k=5]  how many passages to retrieve
 * @param {boolean} [args.strict=false]  refuse when the answer's grounding score is below threshold
 * @param {number} [args.groundingThreshold=0.3]  the strict-mode grounding floor
 * @param {(system:string, user:string)=>Promise<object>} [args.chat]  inject a model call (for tests /
 *   alternate providers); defaults to the built-in OpenAI-compatible completion
 * @returns {Promise<{action:string, rationale:string, inDoctrine:boolean, errored:boolean, note:string,
 *   citations:{source:string, text:string}[], grounding:{grounded:boolean, score:number}}>}
 */
export async function respond({ persona, doctrine, retriever, situation, k = 5, strict = false, groundingThreshold, chat = callModel } = {}) {
  if (!persona || typeof persona !== 'string') throw new TypeError('respond: persona (string) required');
  if (!situation || typeof situation !== 'string') throw new TypeError('respond: situation (string) required');
  if (retriever != null && typeof retriever !== 'function') throw new TypeError('respond: retriever must be a function');
  if (typeof chat !== 'function') throw new TypeError('respond: chat must be a function');
  // A bad k must surface as an error, never silently truncate retrieval to 0 (which masquerades as a
  // legitimate out-of-doctrine refusal).
  if (!Number.isFinite(k) || k < 0) return errorResponse('invalid_k', 'k must be a finite, non-negative number.');

  let retrieve = retriever;
  if (!retrieve) {
    if (doctrine == null) throw new TypeError('respond: provide { doctrine } or { retriever }');
    // An empty doctrine grants nothing — REFUSE (nothing is permitted), do not throw.
    if (Array.isArray(doctrine) && doctrine.length === 0) return refusal('out_of_doctrine');
    retrieve = doctrineRetriever(doctrine); // validates array-ness; throws on a non-array
  }

  const top = await retrieve(situation, k);
  if (!Array.isArray(top) || !top.length) return refusal('out_of_doctrine');

  const ctx = top.map((d, i) => `[${i + 1}] ${d.source ? '(' + d.source + ') ' : ''}${d.text}`).join('\n\n');
  const sys = `You are ${persona}. Act STRICTLY in accordance with the provided doctrine and standard operating procedures — never invent capabilities, weapons, timelines, or procedures not present in it. Cite the passages you are acting on like [1]. If the situation requires something the doctrine does not cover, do NOT fabricate it: set outOfDoctrine true and explain the gap. Output JSON only: {"action":"<what the entity does / decides>","rationale":"<why, per the doctrine, with [n] cites>","used":[1,2],"outOfDoctrine":false,"note":""}`;
  const out = await chat(sys, `Doctrine:\n${ctx}\n\nSituation: ${situation}\n\nRespond as ${persona}.`);
  return groundResponse(out, top, { strict, groundingThreshold });
}

export { tokenize, checkGrounding };
