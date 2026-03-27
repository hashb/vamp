// Node smoke test for the Wasm SIMD planning backend
// Usage after building the module (see README):
//   node scripts/wasm_planning.js

(async () => {
  try {
    const mod = await import('../build-wasm/vamp_planning.mjs');
    const instance = await mod.default();

    // Test planning API
    const plan = instance.cwrap('vamp_plan', 'number', []);
    const getPathDof = instance.cwrap('vamp_get_path_dof', 'number', []);
    const getPathData = instance.cwrap('vamp_get_path_data', 'number', []);
    const getObstacleCount = instance.cwrap('vamp_get_obstacle_count', 'number', []);
    const getRobotSphereCount = instance.cwrap('vamp_get_robot_sphere_count', 'number', []);
    const computeFk = instance.cwrap('vamp_compute_fk', 'number',
      ['number', 'number', 'number', 'number', 'number', 'number', 'number']);

    console.log('Obstacles:', getObstacleCount());
    console.log('Robot spheres:', getRobotSphereCount());
    console.log('DOF:', getPathDof());

    // Test FK
    const fkPtr = computeFk(0, -0.785, 0, -2.356, 0, 1.571, 0.785);
    const fkData = new Float32Array(instance.HEAPF32.buffer, fkPtr, getRobotSphereCount() * 4);
    console.log('FK sphere 0: x=' + fkData[0].toFixed(3) + ' y=' + fkData[1].toFixed(3) + ' z=' + fkData[2].toFixed(3) + ' r=' + fkData[3].toFixed(3));

    // Run planning
    const t0 = performance.now();
    const pathLen = plan();
    const elapsed = (performance.now() - t0).toFixed(1);

    if (pathLen > 0) {
      const dof = getPathDof();
      const ptr = getPathData();
      const pathData = new Float32Array(instance.HEAPF32.buffer, ptr, pathLen * dof);
      console.log('OK path_length=' + pathLen + ' time=' + elapsed + 'ms');
      console.log('First config:', Array.from(pathData.slice(0, dof)).map(v => v.toFixed(3)).join(', '));
      console.log('Last config:', Array.from(pathData.slice((pathLen - 1) * dof, pathLen * dof)).map(v => v.toFixed(3)).join(', '));
    } else {
      console.log('Planning failed - no path found');
      process.exit(1);
    }
  } catch (e) {
    console.error('Planning test failed:', e);
    process.exit(1);
  }
})();
