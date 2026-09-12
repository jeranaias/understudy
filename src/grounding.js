// Shared, zero-dependency text utilities and the groundedness proxy. Kept in one place so the agent
// (src/understudy.js) and the V&V harness (src/benchmark.js) tokenize and measure grounding the same
// way — the leash and the proof must agree on what "grounded in the doctrine" means.

const STOP = new Set(
  'a an the of to and or in on for with by is are be as at from that this it its into their'.split(' '),
);

/**
 * Tokenize text into a bag of lowercased content words (drops punctuation, short words, stopwords).
 * Unicode-aware: keeps letters and numbers from any script (`\p{L}\p{N}`), so non-English doctrine /
 * SOPs (Arabic, Cyrillic, accented Latin, …) tokenize instead of being stripped to noise.
 * @param {string} s
 * @returns {string[]}
 */
export function tokenize(s) {
  return String(s || '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));
}

/**
 * Conservative, deterministic English stemmer — collapses common inflections (plurals, -ing/-ed) so
 * "convoys"/"convoy" and "moves"/"move" count as the same evidence. Pure; non-English words with no
 * matching suffix pass through untouched, so it never corrupts a token it doesn't recognize.
 * @param {string} w
 * @returns {string}
 */
export function stem(w) {
  if (w.length <= 3) return w;
  if (w.endsWith('ies') && w.length > 4) return w.slice(0, -3) + 'y';
  if (w.endsWith('ing') && w.length > 5) return w.slice(0, -3);
  if (w.endsWith('ed') && w.length > 4) return w.slice(0, -2);
  if (/(ss|ch|sh|x|z|s)es$/.test(w) && w.length > 4) return w.slice(0, -2); // -es only after a sibilant
  if (w.endsWith('s') && !w.endsWith('ss') && w.length > 3) return w.slice(0, -1); // plain plural / 3rd-person -s
  return w;
}

/** Tokenize into a Set (deduped). @param {string} s @returns {Set<string>} */
export const tokenSet = (s) => new Set(tokenize(s));

/** Tokenize + stem into a Set (deduped) — the grounding vocabulary. @param {string} s @returns {Set<string>} */
export const stemSet = (s) => new Set(tokenize(s).map(stem));

/**
 * Groundedness proxy — the fraction of a response's content words that also appear in the doctrine it
 * cited. Pure and dependency-free; a cheap guard against a response that cites doctrine but wanders
 * beyond it. Not a semantic check — it will not catch a full paraphrase — but stemming makes it
 * tolerant of inflection (plural/tense), and it reliably flags drift into vocabulary the cited
 * passages never used.
 * @param {string} responseText  the response to measure
 * @param {(string|{text:string})[]} citations  the passages the response claims to rest on
 * @param {{ threshold?: number }} [opts]  overlap fraction to count as grounded (default 0.3)
 * @returns {{ grounded: boolean, score: number }}
 */
export function checkGrounding(responseText, citations, opts = {}) {
  const threshold = opts.threshold ?? 0.3;
  const r = stemSet(responseText);
  if (!r.size) return { grounded: false, score: 0 };
  const d = stemSet((citations || []).map((c) => (typeof c === 'string' ? c : c && c.text) || '').join(' '));
  let hit = 0;
  r.forEach((w) => {
    if (d.has(w)) hit++;
  });
  const score = Math.round((hit / r.size) * 100) / 100;
  return { grounded: score >= threshold, score };
}
