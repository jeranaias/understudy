// Shared, zero-dependency text utilities and the groundedness proxy. Kept in one place so the agent
// (src/understudy.js) and the V&V harness (src/benchmark.js) tokenize and measure grounding the same
// way — the leash and the proof must agree on what "grounded in the doctrine" means.

const STOP = new Set(
  'a an the of to and or in on for with by is are be as at from that this it its into their'.split(' '),
);

/**
 * Tokenize text into a bag of lowercased content words (drops punctuation, short words, stopwords).
 * @param {string} s
 * @returns {string[]}
 */
export function tokenize(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[^a-z0-9 ]+/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));
}

/** Tokenize into a Set (deduped). @param {string} s @returns {Set<string>} */
export const tokenSet = (s) => new Set(tokenize(s));

/**
 * Groundedness proxy — the fraction of a response's content words that also appear in the doctrine it
 * cited. Pure and dependency-free; a cheap guard against a response that cites doctrine but wanders
 * beyond it. Not a semantic check — it will not catch paraphrase — but it reliably flags drift into
 * vocabulary the cited passages never used.
 * @param {string} responseText  the response to measure
 * @param {(string|{text:string})[]} citations  the passages the response claims to rest on
 * @param {{ threshold?: number }} [opts]  overlap fraction to count as grounded (default 0.3)
 * @returns {{ grounded: boolean, score: number }}
 */
export function checkGrounding(responseText, citations, opts = {}) {
  const threshold = opts.threshold ?? 0.3;
  const r = tokenSet(responseText);
  if (!r.size) return { grounded: false, score: 0 };
  const d = tokenSet((citations || []).map((c) => (typeof c === 'string' ? c : c && c.text) || '').join(' '));
  let hit = 0;
  r.forEach((w) => {
    if (d.has(w)) hit++;
  });
  const score = Math.round((hit / r.size) * 100) / 100;
  return { grounded: score >= threshold, score };
}
