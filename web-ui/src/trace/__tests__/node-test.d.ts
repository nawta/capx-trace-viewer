/**
 * The parts of Node's built-in test runner and strict assert that the trace
 * viewer's tests use. The web UI has no @types/node (no new dependency), so
 * these declarations let `tsc` check the tests; web-ui/scripts/test-trace.mjs
 * bundles them and runs them with `node --test`.
 */

declare module 'node:test' {
  type TestFn = () => void | Promise<void>;
  export function test(name: string, fn: TestFn): Promise<void>;
  export function describe(name: string, fn: () => void): void;
  export function it(name: string, fn: TestFn): void;
  export default test;
}

declare module 'node:assert/strict' {
  interface StrictAssert {
    (value: unknown, message?: string): void;
    ok(value: unknown, message?: string): void;
    equal(actual: unknown, expected: unknown, message?: string): void;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    notDeepEqual(actual: unknown, expected: unknown, message?: string): void;
    match(value: string, pattern: RegExp, message?: string): void;
    doesNotMatch(value: string, pattern: RegExp, message?: string): void;
    throws(fn: () => unknown, expected?: unknown, message?: string): void;
    rejects(fn: Promise<unknown> | (() => Promise<unknown>), expected?: unknown, message?: string): Promise<void>;
    doesNotThrow(fn: () => unknown, message?: string): void;
  }
  const assert: StrictAssert;
  export default assert;
}
