#!/usr/bin/env node
// Line-delimited JSON bridge for the HeuristicBot pure function.
//
// Usage: node heuristic_runner.js <engine_dir>
//
// Input:  one line of JSON per request, shape {"observation": {...}}
// Output: one line of JSON per response, shape {"action": {...}}
// Errors: one line of JSON with shape {"error": "..."}  then exit(1).
// EOF on stdin closes the process cleanly with exit(0).
// Diagnostics go to stderr, never stdout.

import readline from 'node:readline';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const engineDir = process.argv[2];
if (!engineDir) {
  process.stderr.write('heuristic_runner: missing engine_dir argument\n');
  process.exit(2);
}

const botPath = path.resolve(engineDir, 'src/logic/bots/HeuristicBot.js');
const { HeuristicBot } = await import(pathToFileURL(botPath).href);

const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  try {
    const { observation } = JSON.parse(trimmed);
    if (!observation) throw new Error('missing "observation" field');
    const action = HeuristicBot(observation);
    process.stdout.write(JSON.stringify({ action }) + '\n');
  } catch (err) {
    process.stdout.write(
      JSON.stringify({ error: String((err && err.message) || err) }) + '\n',
    );
    process.exit(1);
  }
});

rl.on('close', () => process.exit(0));
