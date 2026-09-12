// Understudy — an agent that stands in for a scarce subject-matter expert by acting strictly in
// accordance with a body of doctrine / standard operating procedures. It answers or decides AS the
// entity you describe, grounded only in the provided doctrine, and refuses to invent a capability or
// procedure the doctrine doesn't grant. Model calls go to any OpenAI-compatible chat endpoint.
const ENDPOINT = process.env.UNDERSTUDY_ENDPOINT || 'https://openrouter.ai/api/v1/chat/completions';
const MODEL = process.env.UNDERSTUDY_MODEL || 'google/gemini-3-flash-preview';

const STOP = new Set('a an the of to and or in on for with by is are be as at from that this it its into their'.split(' '));
const tokens = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter((w) => w.length > 2 && !STOP.has(w));
const asDoc = (p) => (typeof p === 'string' ? { text: p, source: '' } : { text: p.text, source: p.source || p.cite || '' });

/** Zero-dependency keyword retriever over the doctrine passages. */
export function doctrineRetriever(doctrine) {
  const docs = (doctrine || []).map(asDoc);
  return async (query, k = 5) => {
    const q = new Set(tokens(query));
    return docs
      .map((d) => { const t = new Set(tokens(d.text)); let s = 0; q.forEach((w) => { if (t.has(w)) s++; }); return { ...d, score: q.size ? s / q.size : 0 }; })
      .filter((d) => d.score > 0).sort((a, b) => b.score - a.score).slice(0, k);
  };
}

async function chat(system, user, timeoutMs = 45000) {
  const KEY = process.env.UNDERSTUDY_API_KEY || process.env.OPENROUTER_API_KEY;
  if (!KEY) return { error: 'set UNDERSTUDY_API_KEY (or OPENROUTER_API_KEY)' };
  try {
    const res = await fetch(ENDPOINT, { method: 'POST', signal: AbortSignal.timeout(timeoutMs),
      headers: { Authorization: `Bearer ${KEY}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: MODEL, temperature: 0.2, messages: [{ role: 'system', content: system }, { role: 'user', content: user }] }) });
    if (!res.ok) return { error: `HTTP ${res.status}` };
    const t = (await res.json()).choices?.[0]?.message?.content ?? '';
    const m = t.match(/\{[\s\S]*\}/); if (m) { try { return JSON.parse(m[0]); } catch {} }
    try { return JSON.parse(t); } catch {}
    return { error: 'no-json' };
  } catch (e) { return { error: e?.name === 'AbortError' ? 'timeout' : String(e) }; }
}

/**
 * Have the understudy respond to a situation, acting as `persona`, grounded ONLY in the doctrine.
 * @param {object} args
 * @param {string} args.persona   who the agent is playing (e.g. "an adversary planning cell")
 * @param {(string|{text:string,source?:string})[]} [args.doctrine]  the doctrine/TTP passages
 * @param {(q:string,k:number)=>Promise<{text,source,score}[]>} [args.retriever]  or bring your own
 * @param {string} args.situation  the scenario/prompt to act on
 * @param {number} [args.k=5]
 * @returns {Promise<{action:string, rationale:string, inDoctrine:boolean, note:string,
 *   citations:{source:string,text:string}[]}>}
 */
export async function respond({ persona, doctrine, retriever, situation, k = 5 }) {
  if (!persona || typeof persona !== 'string') throw new TypeError('respond: persona (string) required');
  if (!situation || typeof situation !== 'string') throw new TypeError('respond: situation (string) required');
  const retrieve = retriever || (doctrine?.length ? doctrineRetriever(doctrine) : null);
  if (!retrieve) throw new TypeError('respond: provide { doctrine } or { retriever }');
  const top = await retrieve(situation, k);
  if (!top.length) {
    return { action: 'No action taken.', rationale: 'The doctrine does not address this situation.', inDoctrine: false, note: 'out_of_doctrine', citations: [] };
  }
  const ctx = top.map((d, i) => `[${i + 1}] ${d.source ? '(' + d.source + ') ' : ''}${d.text}`).join('\n\n');
  const sys = `You are ${persona}. Act STRICTLY in accordance with the provided doctrine and standard operating procedures — never invent capabilities, weapons, timelines, or procedures not present in it. Cite the passages you are acting on like [1]. If the situation requires something the doctrine does not cover, do NOT fabricate it: set outOfDoctrine true and explain the gap. Output JSON only: {"action":"<what the entity does / decides>","rationale":"<why, per the doctrine, with [n] cites>","used":[1,2],"outOfDoctrine":false,"note":""}`;
  const out = await chat(sys, `Doctrine:\n${ctx}\n\nSituation: ${situation}\n\nRespond as ${persona}.`);
  if (out.error) return { action: '', rationale: '', inDoctrine: false, note: out.error, citations: [] };
  const used = (out.used || []).map((n) => top[n - 1]).filter(Boolean);
  const cite = used.length ? used : top.slice(0, 2);
  return {
    action: out.action || '',
    rationale: out.rationale || '',
    inDoctrine: !out.outOfDoctrine,
    note: out.note || '',
    citations: cite.map((d) => ({ source: d.source || 'doctrine', text: d.text })),
  };
}
