// Minimal Node smoke test for the Wasm SIMD backend
// Usage after building the module (see README):
//   node scripts/wasm_smoke.js

(async () => {
  try {
    const mod = await import('../build-wasm/vamp_smoke.mjs');
    const instance = await mod.default();
    const smoke = instance.cwrap('vamp_wasm_smoke', 'number', []);
    const res = smoke();
    console.log('OK', Number(res).toFixed(6));
  } catch (e) {
    console.error('Smoke test failed:', e);
    process.exit(1);
  }
})();

