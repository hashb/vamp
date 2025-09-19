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

    // Extra vector math to exercise SIMD paths
    V a(1.0f);
    V b(2.0f);
    float vhsum = (a * b + b).hsum(); // 4 lanes of (1*2+2) = 4 -> sum = 16

    float checksum = vhsum;
    return checksum; // Expected 16.0f if logic holds
}

} // extern "C"

