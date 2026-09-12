#!/usr/bin/env node
// Understudy CLI — run the doctrine-bound agent or its fidelity benchmark from a shell.
//   understudy respond --doctrine sop.txt --persona "a support desk" --situation "..." [--strict] [--k 5]
//   understudy bench cases.json --doctrine sop.txt --persona "a support desk" [--strict] [--json]
// A doctrine file is plain text; passages are separated by blank lines. A cases file is JSON:
//   [{ "id": "1", "situation": "...", "expect": "the doctrinally-correct behavior" }]
// Requires UNDERSTUDY_API_KEY (or OPENROUTER_API_KEY) to reach a model; the report/refusal shapes are
// printed either way.
import { readFileSync } from 'node:fs';
import { respond } from './understudy.js';
import { benchmark } from './benchmark.js';

const USAGE = `understudy — a doctrine-bound stand-in agent, with a fidelity benchmark.

Usage:
  understudy respond --doctrine <file> --persona <who> --situation <text> [--k <n>] [--strict] [--json]
  understudy bench   <cases.json> --doctrine <file> --persona <who> [--k <n>] [--strict] [--json]

Options:
  --doctrine <file>   Plain-text doctrine; passages separated by blank lines.
  --persona <who>     Who the agent plays, e.g. "an adversary logistics cell".
  --situation <text>  (respond) The scenario to act on.
  --k <n>             Passages to retrieve (default 5).
  --strict            Refuse when the answer's grounding is weak.
  --json              Emit raw JSON only.
  -h, --help          Show this help.

Env: UNDERSTUDY_API_KEY (or OPENROUTER_API_KEY), UNDERSTUDY_ENDPOINT, UNDERSTUDY_MODEL.`;

function parseArgs(argv) {
  const flags = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--strict' || a === '--json') flags[a.slice(2)] = true;
    else if (a === '-h' || a === '--help') flags.help = true;
    else if (a.startsWith('--')) flags[a.slice(2)] = argv[++i];
    else flags._.push(a);
  }
  return flags;
}

/** Split a plain-text doctrine file into passages (blank-line separated). @returns {{text:string}[]} */
function parseDoctrine(path) {
  const raw = readFileSync(path, 'utf8');
  return raw
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((text) => ({ text }));
}

function fail(msg) {
  console.error(`error: ${msg}\n`);
  console.error(USAGE);
  process.exit(1);
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  const cmd = flags._[0];
  if (flags.help || !cmd) {
    console.log(USAGE);
    process.exit(flags.help ? 0 : 1);
  }
  if (!flags.doctrine) fail('--doctrine <file> is required');
  if (!flags.persona) fail('--persona <who> is required');
  const doctrine = parseDoctrine(flags.doctrine);
  const k = flags.k ? Number(flags.k) : undefined;

  if (cmd === 'respond') {
    if (!flags.situation) fail('respond: --situation <text> is required');
    const r = await respond({ persona: flags.persona, doctrine, situation: flags.situation, k, strict: !!flags.strict });
    if (flags.json) return console.log(JSON.stringify(r, null, 2));
    console.log(`inDoctrine: ${r.inDoctrine}${r.note ? `  (${r.note})` : ''}   grounding: ${r.grounding.score}`);
    console.log(`\naction:\n  ${r.action}`);
    if (r.rationale) console.log(`\nrationale:\n  ${r.rationale}`);
    if (r.citations.length) console.log(`\ncitations:\n${r.citations.map((c) => `  [${c.source}] ${c.text}`).join('\n')}`);
    return;
  }

  if (cmd === 'bench') {
    const casesPath = flags._[1];
    if (!casesPath) fail('bench: a <cases.json> path is required');
    let cases;
    try {
      cases = JSON.parse(readFileSync(casesPath, 'utf8'));
    } catch (e) {
      fail(`could not read cases JSON: ${e.message}`);
    }
    if (!Array.isArray(cases)) fail('bench: cases file must be a JSON array of { situation, expect }');
    const { report, runs } = await benchmark(cases, { persona: flags.persona, doctrine, k, strict: !!flags.strict });
    if (flags.json) return console.log(JSON.stringify({ report, runs }, null, 2));
    console.log(`fidelity: ${report.fidelity}   conforming: ${report.conforming}/${report.n}`);
    console.log(`grounded rate: ${report.groundedRate}   mean grounding: ${report.meanGrounding}`);
    console.log(`by verdict: ${JSON.stringify(report.byVerdict)}`);
    if (report.failures.length) {
      console.log('\nfailures:');
      for (const f of report.failures) console.log(`  [${f.id ?? '?'}] ${f.verdict} — ${f.reasons || ''}`);
    }
    return;
  }

  fail(`unknown command: ${cmd}`);
}

main().catch((e) => {
  console.error(`error: ${e.message}`);
  process.exit(1);
});
