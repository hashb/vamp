#include <cstdlib>
#include <cstdint>

#include <emscripten/emscripten.h>

#include <vamp/vector.hh>
#include <vamp/collision/environment.hh>
#include <vamp/collision/validity.hh>

extern "C" {

EMSCRIPTEN_KEEPALIVE
float vamp_wasm_smoke()
{
    using V = vamp::FloatVector<4>;

    // Build a tiny environment with one sphere at origin (r=0.5)
    vamp::collision::Environment<V> env;
    env.spheres.emplace_back(V(0.0f), V(0.0f), V(0.0f), V(0.5f));
    env.sort();

    // Two checks: one colliding near origin, one far away
    bool c1 = vamp::sphere_environment_in_collision<V>(env, 0.1f, 0.1f, 0.1f, 0.5f);
    bool c2 = vamp::sphere_environment_in_collision<V>(env, 5.0f, 5.0f, 5.0f, 0.5f);

    // Extra vector math to exercise SIMD paths
    V a(1.0f);
    V b(2.0f);
    float vhsum = (a * b + b).hsum(); // 4 lanes of (1*2+2) = 4 -> sum = 16

    float checksum = (c1 ? 1.0f : 0.0f) + (c2 ? 1.0f : 0.0f) + vhsum;
    return checksum; // Expected 17.0f if logic holds
}

} // extern "C"

