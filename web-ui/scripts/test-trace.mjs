#!/usr/bin/env node
/**
 * Unit tests of the trace viewer's logic (web-ui/src/trace/__tests__).
 *
 * Bundles every *.test.ts / *.test.tsx there with the esbuild that vite
 * already installed (no new dependency), as CommonJS for Node, into a
 * temporary directory, and runs the bundles with Node's built-in test runner
 * (`node --test`). React components are rendered with react-dom/server, so no
 * browser or DOM library is needed.
 *
 * Page tests (*.page.test.ts / *.page.test.tsx) run the viewer page's own
 * hooks and effects (polling, reloads, data-asset loading): they are bundled
 * with `react` and `react/jsx-runtime` resolved to the small hook runtime in
 * __tests__/harness (fakeReact.ts, fakeJsxRuntime.ts), and harness/browser.ts
 * supplies a virtual clock and a fake trace server.
 *
 * Usage (from web-ui/):
 *   npm run test:trace              # all test files
 *   npm run test:trace -- pager     # only files whose name contains "pager"
 */

import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const here = path.dirname(fileURLToPath(import.meta.url));
const webUi = path.resolve(here, '..');
const testDir = path.join(webUi, 'src', 'trace', '__tests__');
const harnessDir = path.join(testDir, 'harness');
const filters = process.argv.slice(2);

const files = readdirSync(testDir)
  .filter((f) => /\.test\.tsx?$/.test(f))
  .filter((f) => filters.length === 0 || filters.some((q) => f.includes(q)))
  .sort();
if (files.length === 0) {
  console.error(`no test files match ${filters.join(', ') || '*.test.ts(x)'} in ${testDir}`);
  process.exit(1);
}
const isPageTest = (f) => /\.page\.test\.tsx?$/.test(f);

/** Resolves react to the hook runtime of the page tests (react-dom is not available there). */
const fakeReact = {
  name: 'fake-react',
  setup(b) {
    b.onResolve({ filter: /^react$/ }, () => ({ path: path.join(harnessDir, 'fakeReact.ts') }));
    b.onResolve({ filter: /^react\/jsx(-dev)?-runtime$/ }, () => ({ path: path.join(harnessDir, 'fakeJsxRuntime.ts') }));
    b.onResolve({ filter: /^react-dom(\/.*)?$/ }, () => ({ errors: [{ text: 'page tests run without react-dom' }] }));
  },
};

const outDir = mkdtempSync(path.join(tmpdir(), 'capx-trace-tests-'));
let status = 1;
try {
  const common = {
    bundle: true,
    platform: 'node',
    format: 'cjs',
    target: 'node18',
    jsx: 'automatic',
    outExtension: { '.js': '.cjs' },
    sourcemap: 'inline',
    define: { 'process.env.NODE_ENV': '"test"' },
    logLevel: 'warning',
  };
  const plain = files.filter((f) => !isPageTest(f));
  const pages = files.filter(isPageTest);
  if (plain.length) {
    await build({ ...common, entryPoints: plain.map((f) => path.join(testDir, f)), outdir: path.join(outDir, 'plain') });
  }
  if (pages.length) {
    await build({ ...common, entryPoints: pages.map((f) => path.join(testDir, f)), outdir: path.join(outDir, 'page'), plugins: [fakeReact] });
  }
  const bundles = ['plain', 'page']
    .flatMap((d) => {
      try {
        return readdirSync(path.join(outDir, d))
          .filter((f) => f.endsWith('.cjs'))
          .map((f) => path.join(outDir, d, f));
      } catch {
        return [];
      }
    })
    .sort();
  const run = spawnSync(process.execPath, ['--enable-source-maps', '--test', ...bundles], { stdio: 'inherit' });
  status = run.status ?? 1;
} finally {
  rmSync(outDir, { recursive: true, force: true });
}
process.exit(status);
