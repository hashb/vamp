#pragma once

#if !(defined(__EMSCRIPTEN__) && defined(__wasm_simd128__))
#error "Wasm SIMD backend requires Emscripten with -msimd128"
#endif

#include <cstdint>
#include <limits>

#include <wasm_simd128.h>

#include <vamp/vector/interface.hh>

namespace vamp
{
    // Distinct wrapper types to differentiate int/float vectors at the type level
    struct wasm_i32x4
    {
        v128_t v;
    };

    struct wasm_f32x4
    {
        v128_t v;
    };

    template <>
    struct SIMDVector<wasm_i32x4>
    {
        using VectorT = wasm_i32x4;
        using ScalarT = int32_t;
        static constexpr std::size_t VectorWidth = 4;
        static constexpr std::size_t Alignment = 16;

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            switch (idx)
            {
                case 0:
                    return wasm_i32x4_extract_lane(v.v, 0);
                case 1:
                    return wasm_i32x4_extract_lane(v.v, 1);
                case 2:
                    return wasm_i32x4_extract_lane(v.v, 2);
                default:
                    return wasm_i32x4_extract_lane(v.v, 3);
            }
        }

        template <unsigned int = 0>
        inline static constexpr auto constant(ScalarT v) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_splat(v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_sub(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_add(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_mul(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            v128_t all1 = wasm_i32x4_splat(-1);
            return VectorT{wasm_v128_xor(l.v, all1)};
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_and(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_or(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_eq(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_not(wasm_i32x4_eq(l.v, r.v))};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_gt(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            v128_t lo = wasm_v128_and(l.v, r.v);
            return wasm_i32x4_bitmask(lo) == 0;
        }

        template <unsigned int = 0>
        inline static auto load(const ScalarT *const i) noexcept -> VectorT
        {
            return VectorT{wasm_v128_load(i)};
        }

        template <unsigned int = 0>
        inline static auto load_unaligned(const ScalarT *const i) noexcept -> VectorT
        {
            return VectorT{wasm_v128_load(i)};
        }

        template <unsigned int = 0>
        inline static auto store(ScalarT *i, VectorT v) noexcept -> void
        {
            wasm_v128_store(i, v.v);
        }

        template <unsigned int = 0>
        inline static auto store_unaligned(ScalarT *i, VectorT v) noexcept -> void
        {
            wasm_v128_store(i, v.v);
        }

        template <unsigned int = 0>
        inline static auto mask(VectorT v) noexcept -> unsigned int
        {
            return wasm_i32x4_bitmask(v.v);
        }

        template <unsigned int = 0>
        inline static constexpr auto blend(VectorT a, VectorT b, VectorT blend_mask) noexcept -> VectorT
        {
            return VectorT{wasm_v128_bitselect(b.v, a.v, blend_mask.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_shl(v.v, static_cast<int>(i))};
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, unsigned int i) noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_shr(v.v, static_cast<int>(i))};
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return VectorT{wasm_i32x4_splat(0)};
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_f32x4>)
            {
                // numeric conversion
                return wasm_f32x4{wasm_f32x4_convert_i32x4(v.v)};
            }
            else if constexpr (std::is_same_v<OtherVectorT, VectorT>)
            {
                return v;
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-to type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_f32x4>)
            {
                // truncating conversion
                return VectorT{wasm_i32x4_trunc_sat_f32x4(v.v)};
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-from type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_f32x4>)
            {
                return wasm_f32x4{v.v};
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-as type!");
            }
        }

        template <typename = void>
        inline static auto gather(VectorT idxs, const ScalarT *base) noexcept -> VectorT
        {
            // No native gather; emulate via lane loads
            int i0 = wasm_i32x4_extract_lane(idxs.v, 0);
            int i1 = wasm_i32x4_extract_lane(idxs.v, 1);
            int i2 = wasm_i32x4_extract_lane(idxs.v, 2);
            int i3 = wasm_i32x4_extract_lane(idxs.v, 3);
            v128_t v = wasm_i32x4_make(base[i0], base[i1], base[i2], base[i3]);
            return VectorT{v};
        }

        template <typename = void>
        inline static auto gather_select(VectorT idxs, VectorT mask, VectorT alternative, const ScalarT *base)
            noexcept -> VectorT
        {
            auto overlay = gather(idxs, base);
            // select overlay where mask true, else alternative
            return VectorT{wasm_v128_bitselect(overlay.v, alternative.v, mask.v)};
        }
    };

    template <>
    struct SIMDVector<wasm_f32x4>
    {
        using VectorT = wasm_f32x4;
        using ScalarT = float;
        static constexpr std::size_t VectorWidth = 4;
        static constexpr std::size_t Alignment = 16;

        template <unsigned int = 0>
        inline static auto constant(ScalarT v) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_splat(v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto constant_int(unsigned int v) noexcept -> VectorT
        {
            // bit-level constant
            v128_t vi = wasm_i32x4_splat(static_cast<int32_t>(v));
            return VectorT{vi};
        }

        template <unsigned int = 0>
        inline static auto load(const ScalarT *const f) noexcept -> VectorT
        {
            return VectorT{wasm_v128_load(f)};
        }

        template <unsigned int = 0>
        inline static auto load_unaligned(const ScalarT *const f) noexcept -> VectorT
        {
            return VectorT{wasm_v128_load(f)};
        }

        template <unsigned int = 0>
        inline static auto store(ScalarT *f, VectorT v) noexcept -> void
        {
            wasm_v128_store(f, v.v);
        }

        template <unsigned int = 0>
        inline static auto store_unaligned(ScalarT *f, VectorT v) noexcept -> void
        {
            wasm_v128_store(f, v.v);
        }

        template <unsigned int = 0>
        inline static auto extract(VectorT v, int idx) noexcept -> ScalarT
        {
            switch (idx)
            {
                case 0:
                    return wasm_f32x4_extract_lane(v.v, 0);
                case 1:
                    return wasm_f32x4_extract_lane(v.v, 1);
                case 2:
                    return wasm_f32x4_extract_lane(v.v, 2);
                default:
                    return wasm_f32x4_extract_lane(v.v, 3);
            }
        }

        template <std::size_t idx>
        inline static constexpr auto broadcast_dispatch(VectorT v) noexcept -> VectorT
        {
            float lane;
            if constexpr (idx == 0)
                lane = wasm_f32x4_extract_lane(v.v, 0);
            else if constexpr (idx == 1)
                lane = wasm_f32x4_extract_lane(v.v, 1);
            else if constexpr (idx == 2)
                lane = wasm_f32x4_extract_lane(v.v, 2);
            else
                lane = wasm_f32x4_extract_lane(v.v, 3);
            return VectorT{wasm_f32x4_splat(lane)};
        }

        template <std::size_t... I>
        inline static constexpr auto
        broadcast_lookup(VectorT v, std::size_t lane, std::index_sequence<I...>) noexcept -> VectorT
        {
            VectorT ret = zero_vector();
            (void)std::initializer_list<int>{(
                lane == I ? (ret = broadcast_dispatch<std::integral_constant<int, I>{}>(v)), 0 : 0)...};
            return ret;
        }

        template <unsigned int = 0>
        inline static constexpr auto broadcast(VectorT v, std::size_t lane) noexcept -> VectorT
        {
            return broadcast_lookup(v, lane, std::make_index_sequence<VectorWidth>());
        }

        template <unsigned int = 0>
        inline static constexpr auto bitneg(VectorT l) noexcept -> VectorT
        {
            v128_t all1 = wasm_i32x4_splat(-1);
            return VectorT{wasm_v128_xor(l.v, all1)};
        }

        template <unsigned int = 0>
        inline static constexpr auto neg(VectorT l) noexcept -> VectorT
        {
            v128_t sign = wasm_i32x4_splat(0x80000000);
            return VectorT{wasm_v128_xor(l.v, sign)};
        }

        template <unsigned int = 0>
        inline static constexpr auto add(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_add(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto sub(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_sub(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto mul(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_mul(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_le(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_less_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_lt(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_ge(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_greater_than(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_gt(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_eq(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto cmp_not_equal(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_not(wasm_f32x4_eq(l.v, r.v))};
        }

        template <unsigned int = 0>
        inline static auto floor(VectorT v) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_floor(v.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto div(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_div(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto rcp(VectorT l) noexcept -> VectorT
        {
            // No native fast rcp; use division by 1
            return VectorT{wasm_f32x4_div(wasm_f32x4_splat(1.0f), l.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto mask(VectorT v) noexcept -> unsigned int
        {
            // mask of sign bits; acceptable for boolean masks
            return wasm_i32x4_bitmask(v.v);
        }

        template <unsigned int = 0>
        inline static auto zero_vector() noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_splat(0.0f)};
        }

        template <unsigned int = 0>
        inline static auto test_zero(VectorT l, VectorT r) noexcept -> unsigned int
        {
            v128_t anded = wasm_v128_and(l.v, r.v);
            return wasm_i32x4_bitmask(anded) == 0;
        }

        template <unsigned int = 0>
        inline static constexpr auto abs(VectorT v) noexcept -> VectorT
        {
            v128_t mask = wasm_i32x4_splat(0x7fffffff);
            return VectorT{wasm_v128_and(v.v, mask)};
        }

        template <unsigned int = 0>
        inline static constexpr auto and_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_and(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto or_(VectorT l, VectorT r) noexcept -> VectorT
        {
            return VectorT{wasm_v128_or(l.v, r.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto sqrt(VectorT v) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_sqrt(v.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_left(VectorT v, unsigned int i) noexcept -> VectorT
        {
            v128_t vi = wasm_i32x4_shl(v.v, static_cast<int>(i));
            return VectorT{vi};
        }

        template <unsigned int = 0>
        inline static constexpr auto shift_right(VectorT v, unsigned int i) noexcept -> VectorT
        {
            v128_t vi = wasm_i32x4_shr(v.v, static_cast<int>(i));
            return VectorT{vi};
        }

        template <unsigned int = 0>
        inline static constexpr auto clamp(VectorT v, VectorT lower, VectorT upper) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_min(wasm_f32x4_max(v.v, lower.v), upper.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto max(VectorT v, VectorT other) noexcept -> VectorT
        {
            return VectorT{wasm_f32x4_max(v.v, other.v)};
        }

        template <unsigned int = 0>
        inline static constexpr auto hsum(VectorT v) noexcept -> ScalarT
        {
            float out[4];
            wasm_v128_store(out, v.v);
            return out[0] + out[1] + out[2] + out[3];
        }

        // Ported from AVX/NEON versions
        template <unsigned int = 0>
        inline static auto log(VectorT x) noexcept -> VectorT
        {
            using IntVector = SIMDVector<wasm_i32x4>;

            const auto half = constant(0.5F);
            const auto one = constant(1.0F);
            auto invalid_mask = cmp_less_equal(x, zero_vector());

            // Cut off denormalized values
            x = max(x, constant_int(0x00800000));

            auto emm0 = IntVector::shift_right(as<IntVector::VectorT>(x), 23);

            x = and_(x, constant_int(~0x7f800000));
            x = or_(x, half);

            // Keep only the fractional part
            emm0 = IntVector::sub(emm0, IntVector::constant(0x7f));
            auto e = from<IntVector::VectorT>(emm0);

            e = add(e, one);

            // Compute approx
            auto mask = cmp_less_than(x, constant(0.707106781186547524f));
            auto tmp = and_(x, mask);
            x = sub(x, one);
            e = sub(e, and_(one, mask));
            x = add(x, tmp);

            auto z = mul(x, x);

            auto y = constant(7.0376836292E-2f);
            y = mul(y, x);
            y = add(y, constant(-1.1514610310E-1f));
            y = mul(y, x);
            y = add(y, constant(1.1676998740E-1f));
            y = mul(y, x);
            y = add(y, constant(-1.2420140846E-1f));
            y = mul(y, x);
            y = add(y, constant(+1.4249322787E-1f));
            y = mul(y, x);
            y = add(y, constant(-1.6668057665E-1f));
            y = mul(y, x);
            y = add(y, constant(+2.0000714765E-1f));
            y = mul(y, x);
            y = add(y, constant(-2.4999993993E-1f));
            y = mul(y, x);
            y = add(y, constant(+3.3333331174E-1f));
            y = mul(y, mul(x, z));
            tmp = mul(e, constant(-2.12194440e-4f));
            y = add(y, tmp);
            tmp = mul(z, half);
            y = sub(y, tmp);
            tmp = mul(e, constant(0.693359375f));
            x = add(x, add(y, tmp));

            x = or_(x, invalid_mask);  // negative arg will be NAN
            return x;
        }

        // Port of AVX/NEON sin implementation
        template <unsigned int = 0>
        inline static constexpr auto sin(VectorT x) noexcept -> VectorT
        {
            using IntVector = SIMDVector<wasm_i32x4>;

            const auto ps_sign_mask = constant_int(0x80000000);
            const auto ps_cephes_FOPI = constant(1.27323954473516f);  // 4/Pi
            const auto pi32_1 = IntVector::constant(1);
            const auto pi32_inv1 = IntVector::constant(~1);
            const auto pi32_4 = IntVector::constant(4);
            const auto pi32_2 = IntVector::constant(2);
            const auto ps_minus_cephes_DP1 = constant(-0.78515625f);
            const auto ps_minus_cephes_DP2 = constant(-2.4187564849853515625e-4f);
            const auto ps_minus_cephes_DP3 = constant(-3.77489497744594108e-8f);
            const auto ps_coscof_p0 = constant(2.443315711809948E-005f);
            const auto ps_coscof_p1 = constant(-1.388731625493765E-003f);
            const auto ps_coscof_p2 = constant(4.166664568298827E-002f);
            const auto ps_0p5 = constant(0.5f);
            const auto ps_1 = constant(1.0f);
            const auto ps_sincof_p0 = constant(-1.9515295891E-4f);
            const auto ps_sincof_p1 = constant(8.3321608736E-3f);
            const auto ps_sincof_p2 = constant(-1.6666654611E-1f);

            auto sign_bit = x;
            x = abs(x);
            sign_bit = and_(sign_bit, ps_sign_mask);
            auto y = mul(x, ps_cephes_FOPI);

            auto emm2 = IntVector::from(y);

            // j=(j+1) & (~1)
            emm2 = IntVector::add(emm2, pi32_1);
            emm2 = IntVector::and_(emm2, pi32_inv1);
            y = from<typename IntVector::VectorT>(emm2);

            // Get the swap sign flag
            auto emm0 = IntVector::and_(emm2, pi32_4);
            emm0 = IntVector::shift_left(emm0, 29);

            // Get the polynom selection mask
            auto emm2_and2 = IntVector::and_(emm2, pi32_2);
            auto emm2_is_zero = IntVector::cmp_equal(emm2_and2, IntVector::zero_vector());
            auto swap_sign_bit = IntVector::template as<VectorT>(emm0);
            auto poly_mask = IntVector::template as<VectorT>(emm2_is_zero);
            sign_bit = VectorT{wasm_v128_xor(sign_bit.v, swap_sign_bit.v)};

            // The magic pass: Extended precision modular arithmetic
            auto xmm1 = mul(y, ps_minus_cephes_DP1);
            auto xmm2f = mul(y, ps_minus_cephes_DP2);
            auto xmm3 = mul(y, ps_minus_cephes_DP3);
            x = add(x, xmm1);
            x = add(x, xmm2f);
            x = add(x, xmm3);

            // Evaluate the first polynom (cosine)
            auto z = mul(x, x);
            auto y_cos = ps_coscof_p0;
            y_cos = mul(y_cos, z);
            y_cos = add(y_cos, ps_coscof_p1);
            y_cos = mul(y_cos, z);
            y_cos = add(y_cos, ps_coscof_p2);
            y_cos = mul(y_cos, z);
            y_cos = mul(y_cos, z);
            auto tmp = mul(z, ps_0p5);
            y_cos = sub(y_cos, tmp);
            y_cos = add(y_cos, ps_1);

            // Evaluate the second polynom (sine)
            auto y_sin = ps_sincof_p0;
            y_sin = mul(y_sin, z);
            y_sin = add(y_sin, ps_sincof_p1);
            y_sin = mul(y_sin, z);
            y_sin = add(y_sin, ps_sincof_p2);
            y_sin = mul(y_sin, z);
            y_sin = mul(y_sin, x);
            y_sin = add(y_sin, x);

            // Select the correct result from the two polynoms
            auto y2_sel = and_(poly_mask, y_sin);
            auto y1_sel = VectorT{wasm_v128_and(wasm_v128_not(poly_mask.v), y_cos.v)};
            auto y_final = add(y1_sel, y2_sel);

            // Update the sign
            y_final = VectorT{wasm_v128_xor(y_final.v, sign_bit.v)};
            return y_final;
        }

        template <unsigned int = 0>
        inline static constexpr auto blend(VectorT a, VectorT b, VectorT blend_mask) noexcept -> VectorT
        {
            // choose b where mask is true, else a
            return VectorT{wasm_v128_bitselect(b.v, a.v, blend_mask.v)};
        }

        template <unsigned int blend_mask>
        inline static constexpr auto blend_constant(VectorT a, VectorT b) noexcept -> VectorT
        {
            // No immediate-mask blend; emulate by materializing mask and using bitselect
            // Only used with a few known masks in trim/pack_and_pad; fall back to per-bit mask
            constexpr unsigned m = blend_mask & 0xF;
            v128_t mask = wasm_i32x4_make(
                (m & 0x1) ? -1 : 0, (m & 0x2) ? -1 : 0, (m & 0x4) ? -1 : 0, (m & 0x8) ? -1 : 0);
            return VectorT{wasm_v128_bitselect(b.v, a.v, mask)};
        }

        template <typename OtherVectorT>
        inline static constexpr auto to(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_i32x4>)
            {
                // float -> int conversion
                return wasm_i32x4{wasm_i32x4_trunc_sat_f32x4(v.v)};
            }
            else if constexpr (std::is_same_v<OtherVectorT, VectorT>)
            {
                return v;
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-to type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto from(OtherVectorT v) noexcept -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_i32x4>)
            {
                // int -> float conversion
                return VectorT{wasm_f32x4_convert_i32x4(v.v)};
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-from type!");
            }
        }

        template <typename OtherVectorT>
        inline static constexpr auto as(VectorT v) noexcept -> OtherVectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_i32x4>)
            {
                return wasm_i32x4{v.v};
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid cast-as type!");
            }
        }

        template <typename OtherVectorT>
        inline static auto map_to_range(OtherVectorT v) -> VectorT
        {
            if constexpr (std::is_same_v<OtherVectorT, wasm_i32x4>)
            {
                // Map [0, UINT_MAX] to [0, 1]
                auto v1 = wasm_i32x4_and(v.v, wasm_i32x4_splat(1));
                auto v1f = wasm_f32x4_convert_i32x4(v1);
                auto vf = wasm_f32x4_convert_i32x4(v.v);
                auto vscaled = wasm_f32x4_add(vf, v1f);
                return VectorT{wasm_f32x4_mul(
                    vscaled,
                    wasm_f32x4_splat(1.F / static_cast<float>(std::numeric_limits<unsigned int>::max())))};
            }
            else
            {
                static_assert(!sizeof(OtherVectorT), "Invalid range-map type!");
            }
        }

        template <typename = void>
        inline static auto gather(wasm_i32x4 idxs, const ScalarT *base) noexcept -> VectorT
        {
            int i0 = wasm_i32x4_extract_lane(idxs.v, 0);
            int i1 = wasm_i32x4_extract_lane(idxs.v, 1);
            int i2 = wasm_i32x4_extract_lane(idxs.v, 2);
            int i3 = wasm_i32x4_extract_lane(idxs.v, 3);
            v128_t v = wasm_f32x4_make(base[i0], base[i1], base[i2], base[i3]);
            return VectorT{v};
        }

        template <typename = void>
        inline static auto
        gather_select(wasm_i32x4 idxs, VectorT mask, VectorT alternative, const ScalarT *base) noexcept
            -> VectorT
        {
            auto overlay = gather(idxs, base);
            return VectorT{wasm_v128_bitselect(overlay.v, alternative.v, mask.v)};
        }
    };
}  // namespace vamp

