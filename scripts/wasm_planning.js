// Minimal Node smoke test for the Wasm SIMD backend
// Usage after building the module (see README):
//   node scripts/wasm_planning.js

(async () => {
  try {
    const mod = await import('../build-wasm/vamp_planning.mjs');
    const instance = await mod.default();
    const smoke = instance.cwrap('vamp_wasm_planning', 'number', []);
    const res = smoke();
    console.log('OK', Number(res).toFixed(6));
  } catch (e) {
    console.error('Planning test failed:', e);
    process.exit(1);
  }
})();

