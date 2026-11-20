//! Unpacking squeezed (quantized) weights back into normal f32 numbers.

/// Half-precision (16-bit) decimal -> normal 32-bit decimal, by hand.
/// f16 bits: 1 sign | 5 exponent | 10 fraction
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let fraction = (bits & 0x3FF) as u32;
    let f32_bits = if exponent == 0 {
        if fraction == 0 {
            sign // +0 or -0
        } else {
            // Very small ("subnormal") number: fraction × 2^-24
            let value = fraction as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -value } else { value };
        }
    } else if exponent == 31 {
        sign | 0x7F80_0000 | (fraction << 13) // infinity or NaN
    } else {
        // Normal number: f16 stores exponents +15, f32 stores them +127, so add 112
        sign | ((exponent + 112) << 23) | (fraction << 13)
    };
    f32::from_bits(f32_bits)
}

/// Q4_0: blocks of 32 numbers in 18 bytes = a 2-byte f16 scale + 16 bytes of 4-bit values.
/// number = (4-bit value - 8) × scale
pub fn dequantize_q4_0(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(18) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let packed = &block[2..18];
        // Low halves of the 16 bytes are numbers 0..16
        for &byte in packed {
            out.push(((byte & 0x0F) as i32 - 8) as f32 * scale);
        }
        // High halves are numbers 16..32
        for &byte in packed {
            out.push(((byte >> 4) as i32 - 8) as f32 * scale);
        }
    }
} 

/// Q6_K: blocks of 256 numbers in 210 bytes:
///   ql[128] low 4 bits · qh[64] high 2 bits · scales[16] (one per 16 numbers) · d (f16)
/// number = d × scale × (6-bit value - 32)
pub fn dequantize_q6_k(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(210) {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let mut ql = &block[0..128];
        let mut qh = &block[128..192];
        let mut scales = &block[192..208];
        let mut y = [0f32; 256];
        // The block is done in two halves of 128 numbers
        for half in 0..2 {
            let y = &mut y[half * 128..half * 128 + 128];
            for l in 0..32 {
                let group = l / 16;
                // Each qh byte holds the high 2 bits for FOUR numbers (l, l+32, l+64, l+96)
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                let s = |i: usize| scales[i] as i8 as f32; // scales are signed
                y[l] = d * s(group) * q1 as f32;
                y[l + 32] = d * s(group + 2) * q2 as f32;
                y[l + 64] = d * s(group + 4) * q3 as f32;
                y[l + 96] = d * s(group + 6) * q4 as f32;
            }
            // Move on to the second half's bytes
            ql = &ql[64..];
            qh = &qh[32..];
            scales = &scales[8..];
        }
        out.extend_from_slice(&y);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0); // the biggest f16
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x0001), 1.0 / 16_777_216.0); // the smallest f16
        assert!(f16_to_f32(0x7C00).is_infinite());
    }
        #[test]
    fn q4_0_one_block() {
        // scale = 0.5 (f16 0x3800), first byte 0x9C, every other byte 0x88 (= value 8 → 0.0)
        let mut block = vec![0x00, 0x38, 0x9C];
        block.extend([0x88u8; 15]);
        let mut out = Vec::new();
        dequantize_q4_0(&block, &mut out);
        assert_eq!(out.len(), 32);
        assert_eq!(out[0], 2.0);  // low half of 0x9C = 12 → (12 - 8) × 0.5
        assert_eq!(out[16], 0.5); // high half of 0x9C = 9 → (9 - 8) × 0.5
        assert_eq!(out[1], 0.0);  // 8 → (8 - 8) × 0.5
    }

        #[test]
    fn q6_k_one_block() {
        let mut block = vec![0u8; 210];
        for s in &mut block[192..208] {
            *s = 1; // every scale = 1
        }
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0 (f16 0x3C00)
        block[0] = 0x25; // ql[0]: low 4 bits 5 (number 0), high 4 bits 2 (number 64)
        block[128] = 0b01; // qh[0]: high 2 bits of number 0 = 1
        block[192 + 2] = 2; // the scale for numbers 32..48 = 2
        let mut out = Vec::new();
        dequantize_q6_k(&block, &mut out);
        assert_eq!(out.len(), 256);
        assert_eq!(out[0], -11.0); // (1 << 4 | 5) = 21 → 21 - 32
        assert_eq!(out[64], -30.0); // 2 - 32
        assert_eq!(out[32], -64.0); // (0 - 32) × scale 2
        assert_eq!(out[200], -32.0); // all-zero bits → 0 - 32
    }
}