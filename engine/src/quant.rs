//! Unpacking squeezed (quantized) weights back into normal f32 numbers,
//! and multiplying with them.

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

/// Every storage type the engine understands: (numbers per block, bytes per block).
/// GGUF type ids: 0 F32 · 1 F16 · 2 Q4_0 · 3 Q4_1 · 6 Q5_0 · 7 Q5_1 · 8 Q8_0 ·
/// 10 Q2_K · 11 Q3_K · 12 Q4_K · 13 Q5_K · 14 Q6_K · 30 BF16
pub fn block_info(kind: u32) -> Option<(usize, usize)> {
    Some(match kind {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0: scale + 32 × 4 bits
        3 => (32, 20),    // Q4_1: scale + min + 32 × 4 bits
        6 => (32, 22),    // Q5_0: scale + 32 fifth bits + 32 × 4 bits
        7 => (32, 24),    // Q5_1: scale + min + fifth bits + 32 × 4 bits
        8 => (32, 34),    // Q8_0: scale + 32 × 8 bits
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        30 => (1, 2),     // BF16
        _ => return None,
    })
}

/// A readable name for a storage type, for error messages.
pub fn kind_name(kind: u32) -> String {
    match kind {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 30 => "BF16",
        _ => return format!("type {kind}"),
    }
    .to_string()
}

/// How many bytes one row of `cols` numbers takes in a storage type (None = not supported).
pub fn row_bytes(kind: u32, cols: usize) -> Option<usize> {
    let (per_block, bytes) = block_info(kind)?;
    (cols % per_block == 0).then(|| cols / per_block * bytes)
}

/// Unpack any supported storage type into normal f32 numbers.
pub fn dequantize(kind: u32, bytes: &[u8], out: &mut Vec<f32>) {
    let half = |b: &[u8]| f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
    match kind {
        0 => out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        1 => out.extend(bytes.chunks_exact(2).map(half)),
        // BF16 is simply the top half of an f32
        30 => out.extend(bytes.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))),
        2 => dequantize_q4_0(bytes, out),
        3 => dequantize_q4_1(bytes, out),
        6 => dequantize_q5_0(bytes, out),
        7 => dequantize_q5_1(bytes, out),
        8 => dequantize_q8_0(bytes, out),
        10 => dequantize_q2_k(bytes, out),
        11 => dequantize_q3_k(bytes, out),
        12 => dequantize_q4_k(bytes, out),
        13 => dequantize_q5_k(bytes, out),
        14 => dequantize_q6_k(bytes, out),
        other => panic!("storage type {other} isn't supported"),
    }
}

fn half_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// Q4_1: like Q4_0, but with a minimum instead of "- 8".  number = value × scale + min
pub fn dequantize_q4_1(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(20) {
        let (d, m) = (half_at(block, 0), half_at(block, 2));
        let packed = &block[4..20];
        out.extend(packed.iter().map(|&b| (b & 0x0F) as f32 * d + m));
        out.extend(packed.iter().map(|&b| (b >> 4) as f32 * d + m));
    }
}

/// Q5_0: 4 bits in qs + a 5th bit for each number in a 32-bit mask.
/// number = (5-bit value - 16) × scale
pub fn dequantize_q5_0(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(22) {
        let d = half_at(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let mut y = [0f32; 32];
        for j in 0..16 {
            let high0 = ((qh >> j) & 1) << 4; // 5th bit of number j
            let high1 = ((qh >> (j + 16)) & 1) << 4; // 5th bit of number j + 16
            y[j] = (((qs[j] & 0x0F) as u32 | high0) as i32 - 16) as f32 * d;
            y[j + 16] = (((qs[j] >> 4) as u32 | high1) as i32 - 16) as f32 * d;
        }
        out.extend_from_slice(&y);
    }
}

/// Q5_1: Q5_0 with a minimum.  number = 5-bit value × scale + min
pub fn dequantize_q5_1(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(24) {
        let (d, m) = (half_at(block, 0), half_at(block, 2));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let mut y = [0f32; 32];
        for j in 0..16 {
            let high0 = ((qh >> j) & 1) << 4;
            let high1 = ((qh >> (j + 16)) & 1) << 4;
            y[j] = ((qs[j] & 0x0F) as u32 | high0) as f32 * d + m;
            y[j + 16] = ((qs[j] >> 4) as u32 | high1) as f32 * d + m;
        }
        out.extend_from_slice(&y);
    }
}

/// Q8_0: one scale + 32 signed bytes.  number = byte × scale
pub fn dequantize_q8_0(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(34) {
        let d = half_at(block, 0);
        out.extend(block[2..34].iter().map(|&q| q as i8 as f32 * d));
    }
}

/// K-quants pack eight 6-bit (scale, min) pairs into 12 bytes. Pair j:
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (
            ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)) as f32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32,
        )
    }
}

/// Q4_K: 256 numbers = 8 groups of 32, each group with its own 6-bit scale and min.
/// Layout: d (f16) · dmin (f16) · scales[12] · qs[128]
/// number = d × scale × 4-bit value - dmin × min
pub fn dequantize_q4_k(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(144) {
        let (d, dmin) = (half_at(block, 0), half_at(block, 2));
        let scales = &block[4..16];
        let mut q = &block[16..144];
        for pair in 0..4 {
            // 64 numbers: low halves of 32 bytes (group 2·pair), then high halves (group 2·pair + 1)
            let (sc1, m1) = scale_min_k4(2 * pair, scales);
            let (sc2, m2) = scale_min_k4(2 * pair + 1, scales);
            out.extend(q[..32].iter().map(|&b| d * sc1 * (b & 0x0F) as f32 - dmin * m1));
            out.extend(q[..32].iter().map(|&b| d * sc2 * (b >> 4) as f32 - dmin * m2));
            q = &q[32..];
        }
    }
}

/// Q5_K: Q4_K plus a 5th bit per number.
/// Layout: d · dmin · scales[12] · qh[32] (5th bits) · qs[128]
pub fn dequantize_q5_k(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(176) {
        let (d, dmin) = (half_at(block, 0), half_at(block, 2));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let mut q = &block[48..176];
        for pair in 0..4 {
            let (sc1, m1) = scale_min_k4(2 * pair, scales);
            let (sc2, m2) = scale_min_k4(2 * pair + 1, scales);
            let (bit1, bit2) = (1u8 << (2 * pair), 1u8 << (2 * pair + 1));
            for l in 0..32 {
                let v = (q[l] & 0x0F) as f32 + if qh[l] & bit1 != 0 { 16.0 } else { 0.0 };
                out.push(d * sc1 * v - dmin * m1);
            }
            for l in 0..32 {
                let v = (q[l] >> 4) as f32 + if qh[l] & bit2 != 0 { 16.0 } else { 0.0 };
                out.push(d * sc2 * v - dmin * m2);
            }
            q = &q[32..];
        }
    }
}

/// Q2_K: 2 bits per number, 16 groups of 16, each with a 4-bit scale and 4-bit min.
/// Layout: scales[16] · qs[64] · d (f16) · dmin (f16)
pub fn dequantize_q2_k(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(84) {
        let scales = &block[0..16];
        let (d, dmin) = (half_at(block, 80), half_at(block, 82));
        let mut is = 0;
        for half in 0..2 {
            let q = &block[16 + half * 32..16 + half * 32 + 32];
            for shift in [0, 2, 4, 6] {
                for part in 0..2 {
                    let sc = scales[is];
                    is += 1;
                    let (dl, ml) = (d * (sc & 0x0F) as f32, dmin * (sc >> 4) as f32);
                    for l in 0..16 {
                        out.push(dl * ((q[part * 16 + l] >> shift) & 3) as f32 - ml);
                    }
                }
            }
        }
    }
}

/// Q3_K: 2 low bits in qs + 1 high bit in hmask, 16 groups of 16 with 6-bit signed scales.
/// Layout: hmask[32] · qs[64] · scales[12] · d (f16)
pub fn dequantize_q3_k(bytes: &[u8], out: &mut Vec<f32>) {
    for block in bytes.chunks_exact(110) {
        let hmask = &block[0..32];
        let d = half_at(block, 108);
        // Unpack the twelve bytes into sixteen 6-bit scales
        let raw = &block[96..108];
        let mut scales = [0i8; 16];
        for (i, s) in scales.iter_mut().enumerate() {
            let low = if i < 8 { raw[i] & 0x0F } else { raw[i - 8] >> 4 };
            let high = (raw[8 + i % 4] >> (2 * (i / 4))) & 3;
            *s = (low | (high << 4)) as i8 - 32;
        }
        let mut is = 0;
        let mut m = 1u8;
        for half in 0..2 {
            let q = &block[32 + half * 32..32 + half * 32 + 32];
            for shift in [0, 2, 4, 6] {
                for part in 0..2 {
                    let dl = d * scales[is] as f32;
                    is += 1;
                    for l in 0..16 {
                        let i = part * 16 + l;
                        let low = ((q[i] >> shift) & 3) as i32;
                        let v = low - if hmask[i] & m != 0 { 0 } else { 4 };
                        out.push(dl * v as f32);
                    }
                }
                m <<= 1;
            }
        }
    }
}

/// Types that are multiplied with WHOLE-NUMBER maths against a word rounded to 8 bits.
fn uses_q8(kind: u32) -> bool {
    matches!(kind, 2 | 6 | 8 | 12) // Q4_0, Q5_0, Q8_0, Q4_K
}

/// A weight matrix kept in its packed form. Rows are unpacked only when needed,
/// so the 1.5B model stays ~1 GB in memory instead of ~6 GB.
pub struct QMatrix {
    pub kind: u32,
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

impl QMatrix {
    /// Unpack one row into `out`.
    pub fn row(&self, r: usize, out: &mut Vec<f32>) {
        let size = row_bytes(self.kind, self.cols).unwrap();
        let bytes = &self.bytes[r * size..(r + 1) * size];
        out.clear();
        dequantize(self.kind, bytes, out);
    }

    /// Many vectors at once: ys[t] = W · xs[t] for every t.
    /// Each weight row is read from RAM ONCE and used for all the vectors while it's
    /// still in the CPU cache. This is what makes reading a prompt fast.
    pub fn matmul(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        use rayon::prelude::*;
        let n = xs.len();
        for x in xs {
            assert_eq!(x.len(), self.cols, "vector size doesn't match the matrix");
        }
        const ROWS_PER_JOB: usize = 16;
        // Round every word to 8-bit blocks once (Q4_0 only)
        let q8s: Vec<Option<Q8Vector>> = xs.iter().map(|x| uses_q8(self.kind).then(|| quantize_q8_0(x))).collect();
        // Results stored row by row: flat[r * n + t] = row r · xs[t]
        let mut flat = vec![0.0; self.rows * n];
        flat.par_chunks_mut(ROWS_PER_JOB * n).enumerate().for_each(|(job, part)| {
            let mut buf = Vec::new();
            for (j, out_row) in part.chunks_mut(n).enumerate() {
                let r = job * ROWS_PER_JOB + j;
                for (t, out) in out_row.iter_mut().enumerate() {
                    *out = self.row_dot(r, &xs[t], q8s[t].as_ref(), &mut buf); // same row bytes, still in cache
                }
            }
        });
        // Turn "row by row" into "one result list per vector"
        (0..n).map(|t| (0..self.rows).map(|r| flat[r * n + t]).collect()).collect()
    }

    /// The old, simple version (kept to compare speed and results).
    pub fn matvec_simple(&self, x: &[f32]) -> Vec<f32> {
        let mut row = Vec::with_capacity(self.cols);
        (0..self.rows)
            .map(|r| {
                self.row(r, &mut row);
                row.iter().zip(x).map(|(w, v)| w * v).sum()
            })
            .collect()
    }

    /// One row · x. Q4_0, Q5_0, Q8_0 and Q4_K rows use integer maths with the rounded word (q8);
    /// Q6_K rows are multiplied straight from their packed bytes; other types are unpacked first.
    fn row_dot(&self, r: usize, x: &[f32], q8: Option<&Q8Vector>, buf: &mut Vec<f32>) -> f32 {
        let size = row_bytes(self.kind, self.cols).unwrap();
        let row = &self.bytes[r * size..(r + 1) * size];
        match (self.kind, q8) {
            (2, Some(q8)) => return dot_q4_0_q8_0(row, q8),
            (6, Some(q8)) => return dot_q5_0_q8_0(row, q8),
            (8, Some(q8)) => return dot_q8_0_q8_0(row, q8),
            (12, Some(q8)) => return dot_q4_k_q8_0(row, q8),
            (2, None) => return dot_q4_0(row, x),
            (14, _) => return dot_q6_k(row, x),
            _ => {}
        }
        self.row(r, buf);
        buf.iter().zip(x).map(|(w, v)| w * v).sum()
    }

    /// y = W · x, with the rows shared out between all CPU cores.
    /// rayon keeps one thread per core alive and hands them work, instead of
    /// starting new threads for every multiplication.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        assert_eq!(x.len(), self.cols, "vector size doesn't match the matrix");
        const ROWS_PER_JOB: usize = 16; // small jobs, so even 256-row matrices keep 16 cores busy
        // Round the word to 8-bit blocks ONCE, then reuse it for every row (Q4_0 only)
        let q8 = uses_q8(self.kind).then(|| quantize_q8_0(x));
        let mut y = vec![0.0; self.rows];
        y.par_chunks_mut(ROWS_PER_JOB).enumerate().for_each(|(job, part)| {
            let mut buf = Vec::new(); // only used by F32 rows
            for (j, out) in part.iter_mut().enumerate() {
                *out = self.row_dot(job * ROWS_PER_JOB + j, x, q8.as_ref(), &mut buf);
            }
        });
        y
    }
}

/// One Q4_0 row · x. Uses AVX2 (8 numbers per instruction) when the CPU has it,
/// otherwise the plain version. The check is done once and remembered by Rust.
pub fn dot_q4_0(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // Safe: we just checked that this CPU has AVX2 and FMA
            return unsafe { dot_q4_0_avx2(row, x) };
        }
    }
    dot_q4_0_scalar(row, x)
}

/// Plain version: one number at a time. Inside a block every number shares one scale,
/// so add up (value - 8) × x first, then multiply by the scale ONCE per block.
pub fn dot_q4_0_scalar(row: &[u8], x: &[f32]) -> f32 {
    let mut total = 0.0;
    for (block, xs) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut sum = 0.0;
        for j in 0..16 {
            let byte = block[2 + j];
            sum += ((byte & 0x0F) as i32 - 8) as f32 * xs[j]; // numbers 0..16
            sum += ((byte >> 4) as i32 - 8) as f32 * xs[j + 16]; // numbers 16..32
        }
        total += sum * scale;
    }
    total
}

/// AVX2 version: the CPU works on 8 numbers at once, in 256-bit "lanes".
/// Per block of 32: unpack 32 four-bit values in a few instructions,
/// turn them into 4 × 8 decimals, and multiply-add them with x 8 at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_avx2(row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let low_4_bits = _mm_set1_epi8(0x0F);
        let eight = _mm_set1_epi8(8);
        let mut total = _mm256_setzero_ps(); // 8 running totals, added up at the end

        for (block, xs) in row.chunks_exact(18).zip(x.chunks_exact(32)) {
            let scale = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([block[0], block[1]])));

            // 16 packed bytes -> 16 low halves (numbers 0..16) and 16 high halves (16..32)
            let packed = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);
            let low = _mm_sub_epi8(_mm_and_si128(packed, low_4_bits), eight); // value - 8
            let high = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed, 4), low_4_bits), eight);

            // Each group of 8 small whole numbers -> 8 decimals, × 8 numbers of x, added up
            let x = xs.as_ptr();
            let to_f32 = |v: __m128i| _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
            let mut sum = _mm256_mul_ps(to_f32(low), _mm256_loadu_ps(x)); // numbers 0..8
            sum = _mm256_fmadd_ps(to_f32(_mm_srli_si128(low, 8)), _mm256_loadu_ps(x.add(8)), sum); // 8..16
            sum = _mm256_fmadd_ps(to_f32(high), _mm256_loadu_ps(x.add(16)), sum); // 16..24
            sum = _mm256_fmadd_ps(to_f32(_mm_srli_si128(high, 8)), _mm256_loadu_ps(x.add(24)), sum); // 24..32

            // total += sum × scale (still 8 lanes at once)
            total = _mm256_fmadd_ps(sum, scale, total);
        }

        // Add the 8 lanes together into one number
        let half = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        let quarter = _mm_add_ps(half, _mm_movehl_ps(half, half));
        let one = _mm_add_ss(quarter, _mm_shuffle_ps(quarter, quarter, 1));
        _mm_cvtss_f32(one)
    }
}

/// Unpack one Q6_K block's 256 six-bit values into small whole numbers (-32..31).
/// Same bit layout as dequantize_q6_k, but no scales and no decimals yet.
fn q6_k_values(block: &[u8], q: &mut [i8; 256]) {
    for half in 0..2 {
        let ql = &block[half * 64..half * 64 + 64];
        let qh = &block[128 + half * 32..128 + half * 32 + 32];
        let q = &mut q[half * 128..half * 128 + 128];
        for l in 0..32 {
            q[l] = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i8 - 32;
            q[l + 32] = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            q[l + 64] = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            q[l + 96] = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
        }
    }
}

/// One Q6_K row · x, straight from the packed bytes.
/// Every 16 numbers share one small scale (scales[n / 16]), and the block shares d.
pub fn dot_q6_k(row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q6_k_avx2(row, x) };
        }
    }
    dot_q6_k_scalar(row, x)
}

pub fn dot_q6_k_scalar(row: &[u8], x: &[f32]) -> f32 {
    let mut q = [0i8; 256];
    let mut total = 0.0;
    for (block, xs) in row.chunks_exact(210).zip(x.chunks_exact(256)) {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        q6_k_values(block, &mut q);
        for group in 0..16 {
            let scale = block[192 + group] as i8 as f32;
            let mut sum = 0.0;
            for i in group * 16..group * 16 + 16 {
                sum += q[i] as f32 * xs[i];
            }
            total += d * scale * sum;
        }
    }
    total
}

/// AVX2 version: 16 numbers per group = 2 × 8 lanes, scale applied once per group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6_k_avx2(row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut q = [0i8; 256];
        let mut total = _mm256_setzero_ps();
        for (block, xs) in row.chunks_exact(210).zip(x.chunks_exact(256)) {
            let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
            q6_k_values(block, &mut q);
            for group in 0..16 {
                let values = _mm_loadu_si128(q.as_ptr().add(group * 16) as *const __m128i); // 16 small numbers
                let x = xs.as_ptr().add(group * 16);
                let first = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(values)); // numbers 0..8
                let second = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(values, 8))); // 8..16
                let mut sum = _mm256_mul_ps(first, _mm256_loadu_ps(x));
                sum = _mm256_fmadd_ps(second, _mm256_loadu_ps(x.add(8)), sum);
                let scale = _mm256_set1_ps(d * block[192 + group] as i8 as f32);
                total = _mm256_fmadd_ps(sum, scale, total);
            }
        }
        let half = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        let quarter = _mm_add_ps(half, _mm_movehl_ps(half, half));
        let one = _mm_add_ss(quarter, _mm_shuffle_ps(quarter, quarter, 1));
        _mm_cvtss_f32(one)
    }
}

// ---------- Step 6.6: integer maths ----------

/// A word rounded to "Q8_0": every 32 numbers share one scale, and each number
/// becomes a small whole number between -127 and 127.
pub struct Q8Vector {
    pub scales: Vec<f32>, // one per block of 32
    pub values: Vec<i8>,  // number ≈ value × scale
}

/// Round a word's numbers into Q8_0 blocks (done once per matrix multiplication).
pub fn quantize_q8_0(x: &[f32]) -> Q8Vector {
    let mut scales = Vec::with_capacity(x.len() / 32);
    let mut values = Vec::with_capacity(x.len());
    for block in x.chunks_exact(32) {
        let biggest = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = biggest / 127.0; // the biggest number becomes ±127
        let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        scales.push(scale);
        values.extend(block.iter().map(|v| (v * inverse).round() as i8));
    }
    Q8Vector { scales, values }
}

/// One Q4_0 row · a Q8_0 word, with WHOLE-NUMBER maths.
/// Per block: weight_scale × word_scale × Σ (weight_int × word_int)
pub fn dot_q4_0_q8_0(row: &[u8], x: &Q8Vector) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q4_0_q8_0_avx2(row, x) };
        }
    }
    dot_q4_0_q8_0_scalar(row, x)
}

/// Plain version: one pair at a time, but all in whole numbers.
pub fn dot_q4_0_q8_0_scalar(row: &[u8], x: &Q8Vector) -> f32 {
    let mut total = 0.0;
    for (b, block) in row.chunks_exact(18).enumerate() {
        let w_scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let xs = &x.values[b * 32..b * 32 + 32];
        let mut sum: i32 = 0;
        for j in 0..16 {
            let byte = block[2 + j];
            sum += ((byte & 0x0F) as i32 - 8) * xs[j] as i32; // numbers 0..16
            sum += ((byte >> 4) as i32 - 8) * xs[j + 16] as i32; // numbers 16..32
        }
        total += w_scale * x.scales[b] * sum as f32; // scales applied ONCE per block
    }
    total
}

/// AVX2 version: 32 whole-number multiplications in one instruction (maddubs).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_q8_0_avx2(row: &[u8], x: &Q8Vector) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let low_4_bits = _mm_set1_epi8(0x0F);
        let eight = _mm256_set1_epi8(8);
        let ones = _mm256_set1_epi16(1);
        let mut total = _mm256_setzero_ps();

        for (b, block) in row.chunks_exact(18).enumerate() {
            // 32 weights as whole numbers -8..7: low halves in the first 16 bytes, high halves in the last 16
            let packed = _mm_loadu_si128(block.as_ptr().add(2) as *const __m128i);
            let low = _mm_and_si128(packed, low_4_bits);
            let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_4_bits);
            let w = _mm256_sub_epi8(_mm256_set_m128i(high, low), eight);

            // 32 word values (-127..127)
            let xv = _mm256_loadu_si256(x.values.as_ptr().add(b * 32) as *const __m256i);

            // maddubs needs one side without a sign: use |w|, and move w's sign onto x
            let w_abs = _mm256_sign_epi8(w, w);
            let x_signed = _mm256_sign_epi8(xv, w);
            let pairs = _mm256_maddubs_epi16(w_abs, x_signed); // 32 products → 16 sums of 2
            let quads = _mm256_madd_epi16(pairs, ones); //          16 sums → 8 sums of 4 (i32)

            // × both scales, once per block, into the 8 running totals
            let scale = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([block[0], block[1]])) * x.scales[b]);
            total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(quads), scale, total);
        }

        // Add the 8 lanes together into one number
        let half = _mm_add_ps(_mm256_castps256_ps128(total), _mm256_extractf128_ps(total, 1));
        let quarter = _mm_add_ps(half, _mm_movehl_ps(half, half));
        let one = _mm_add_ss(quarter, _mm_shuffle_ps(quarter, quarter, 1));
        _mm_cvtss_f32(one)
    }
}


// ---------- Whole-number dots for the formats in "Q4_K_M" models: Q5_0, Q8_0, Q4_K ----------

/// Q5_0 row · Q8_0 word. Weights are whole numbers -16..15.
pub fn dot_q5_0_q8_0(row: &[u8], x: &Q8Vector) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q5_0_q8_0_avx2(row, x) };
        }
    }
    dot_q5_0_q8_0_scalar(row, x)
}

pub fn dot_q5_0_q8_0_scalar(row: &[u8], x: &Q8Vector) -> f32 {
    let mut total = 0.0;
    for (b, block) in row.chunks_exact(22).enumerate() {
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let xs = &x.values[b * 32..b * 32 + 32];
        let mut sum: i32 = 0;
        for j in 0..16 {
            let byte = block[6 + j];
            let w0 = ((byte & 0x0F) as i32 | (((qh >> j) & 1) << 4) as i32) - 16;
            let w1 = ((byte >> 4) as i32 | (((qh >> (j + 16)) & 1) << 4) as i32) - 16;
            sum += w0 * xs[j] as i32 + w1 * xs[j + 16] as i32;
        }
        total += half_at(block, 0) * x.scales[b] * sum as f32;
    }
    total
}

/// Q8_0 row · Q8_0 word: both sides are already bytes.
pub fn dot_q8_0_q8_0(row: &[u8], x: &Q8Vector) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q8_0_q8_0_avx2(row, x) };
        }
    }
    dot_q8_0_q8_0_scalar(row, x)
}

pub fn dot_q8_0_q8_0_scalar(row: &[u8], x: &Q8Vector) -> f32 {
    let mut total = 0.0;
    for (b, block) in row.chunks_exact(34).enumerate() {
        let xs = &x.values[b * 32..b * 32 + 32];
        let sum: i32 = block[2..34].iter().zip(xs).map(|(&w, &v)| w as i8 as i32 * v as i32).sum();
        total += half_at(block, 0) * x.scales[b] * sum as f32;
    }
    total
}

/// Q4_K row · Q8_0 word. A Q4_K group of 32 lines up exactly with one Q8_0 block, so per group:
///   d·scale·word_scale·Σ(w·x)  -  dmin·min·word_scale·Σx
pub fn dot_q4_k_q8_0(row: &[u8], x: &Q8Vector) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q4_k_q8_0_avx2(row, x) };
        }
    }
    dot_q4_k_q8_0_scalar(row, x)
}

pub fn dot_q4_k_q8_0_scalar(row: &[u8], x: &Q8Vector) -> f32 {
    let mut total = 0.0;
    for (b, block) in row.chunks_exact(144).enumerate() {
        let (d, dmin) = (half_at(block, 0), half_at(block, 2));
        let scales = &block[4..16];
        for g in 0..8 {
            let q = &block[16 + (g / 2) * 32..16 + (g / 2) * 32 + 32];
            let shift = if g % 2 == 0 { 0 } else { 4 };
            let k = b * 8 + g; // which Q8_0 block of the word
            let xs = &x.values[k * 32..k * 32 + 32];
            let mut dot: i32 = 0;
            let mut xsum: i32 = 0;
            for l in 0..32 {
                dot += ((q[l] >> shift) & 0x0F) as i32 * xs[l] as i32;
                xsum += xs[l] as i32;
            }
            let (sc, m) = scale_min_k4(g, scales);
            total += x.scales[k] * (d * sc * dot as f32 - dmin * m * xsum as f32);
        }
    }
    total
}

/// Add up 8 lanes into one number.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_lanes(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let half = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let quarter = _mm_add_ps(half, _mm_movehl_ps(half, half));
    _mm_cvtss_f32(_mm_add_ss(quarter, _mm_shuffle_ps(quarter, quarter, 1)))
}

/// 32 signed weight bytes · 32 signed word bytes → 8 partial sums (i32).
/// maddubs wants one side without a sign: use |w| and move w's sign onto x.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_32_signed(w: std::arch::x86_64::__m256i, x: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))
}

/// The 32 bits of `bits` spread into 32 bytes: 0xFF where the bit is 1, 0 where it is 0.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bits_to_bytes(bits: u32) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // Byte j gets the byte of `bits` that holds bit j ...
    let spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32(bits as i32),
        _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0),
    );
    // ... then every OTHER bit is forced to 1, so the byte is all-ones only if bit j was 1
    let others = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    _mm256_cmpeq_epi8(_mm256_or_si256(spread, others), _mm256_set1_epi64x(-1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5_0_q8_0_avx2(row: &[u8], x: &Q8Vector) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let low_4_bits = _mm_set1_epi8(0x0F);
        let sixteen = _mm256_set1_epi8(16);
        let mut total = _mm256_setzero_ps();
        for (b, block) in row.chunks_exact(22).enumerate() {
            let packed = _mm_loadu_si128(block.as_ptr().add(6) as *const __m128i);
            let low = _mm_and_si128(packed, low_4_bits);
            let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_4_bits);
            let nibbles = _mm256_set_m128i(high, low);
            // 5th bits: +16 where set, then -16 for everyone → -16..15
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let fifth = _mm256_and_si256(bits_to_bytes(qh), sixteen);
            let w = _mm256_sub_epi8(_mm256_or_si256(nibbles, fifth), sixteen);
            let xv = _mm256_loadu_si256(x.values.as_ptr().add(b * 32) as *const __m256i);
            let scale = _mm256_set1_ps(half_at(block, 0) * x.scales[b]);
            total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_32_signed(w, xv)), scale, total);
        }
        add_lanes(total)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8_0_avx2(row: &[u8], x: &Q8Vector) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut total = _mm256_setzero_ps();
        for (b, block) in row.chunks_exact(34).enumerate() {
            let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
            let xv = _mm256_loadu_si256(x.values.as_ptr().add(b * 32) as *const __m256i);
            let scale = _mm256_set1_ps(half_at(block, 0) * x.scales[b]);
            total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_32_signed(w, xv)), scale, total);
        }
        add_lanes(total)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_k_q8_0_avx2(row: &[u8], x: &Q8Vector) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let low_4_bits = _mm256_set1_epi8(0x0F);
        let ones_u8 = _mm256_set1_epi8(1);
        let ones_i16 = _mm256_set1_epi16(1);
        let mut total = _mm256_setzero_ps();
        for (b, block) in row.chunks_exact(144).enumerate() {
            let (d, dmin) = (half_at(block, 0), half_at(block, 2));
            let scales = &block[4..16];
            for pair in 0..4 {
                // 32 bytes = group 2·pair (low halves) and group 2·pair+1 (high halves)
                let packed = _mm256_loadu_si256(block.as_ptr().add(16 + pair * 32) as *const __m256i);
                let lows = _mm256_and_si256(packed, low_4_bits);
                let highs = _mm256_and_si256(_mm256_srli_epi16(packed, 4), low_4_bits);
                for (half, w) in [(0, lows), (1, highs)] {
                    let g = pair * 2 + half;
                    let k = b * 8 + g;
                    let xv = _mm256_loadu_si256(x.values.as_ptr().add(k * 32) as *const __m256i);
                    // Weights are 0..15 (no sign), so maddubs can use them directly
                    let dot = _mm256_madd_epi16(_mm256_maddubs_epi16(w, xv), ones_i16);
                    let xsum = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, xv), ones_i16);
                    let (sc, m) = scale_min_k4(g, scales);
                    let xs = x.scales[k];
                    total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(d * sc * xs), total);
                    total = _mm256_fmadd_ps(_mm256_cvtepi32_ps(xsum), _mm256_set1_ps(-dmin * m * xs), total);
                }
            }
        }
        add_lanes(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q5_0, Q8_0 and Q4_K: the whole-number dots (plain AND fast) must match "unpack, then multiply".
    #[test]
    fn new_integer_dots_match_unpack_then_dot() {
        let mut seed = 12345u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        let cols = 512;
        let x: Vec<f32> = (0..cols).map(|_| (rand() % 2001) as f32 / 1000.0 - 1.0).collect();
        let q8 = quantize_q8_0(&x);
        type Dot = fn(&[u8], &Q8Vector) -> f32;
        let cases: [(u32, &[usize], Dot, Dot); 3] = [
            (6, &[0], dot_q5_0_q8_0_scalar, dot_q5_0_q8_0),
            (8, &[0], dot_q8_0_q8_0_scalar, dot_q8_0_q8_0),
            (12, &[0, 2], dot_q4_k_q8_0_scalar, dot_q4_k_q8_0),
        ];
        for (kind, halves, plain, fast) in cases {
            let (per_block, block_bytes) = block_info(kind).unwrap();
            let mut row: Vec<u8> = (0..cols / per_block * block_bytes).map(|_| rand() as u8).collect();
            for block in row.chunks_exact_mut(block_bytes) {
                for &off in halves {
                    block[off..off + 2].copy_from_slice(&0x2000u16.to_le_bytes()); // f16 ≈ 0.0078
                }
            }
            let mut w = Vec::new();
            dequantize(kind, &row, &mut w);
            let exact: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();
            let size: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
            let (p, f) = (plain(&row, &q8), fast(&row, &q8));
            // The word was rounded to 8 bits, so allow ~1% of the total size
            assert!((p - exact).abs() <= 0.01 * size, "{}: plain {p} vs exact {exact}", kind_name(kind));
            assert!((f - p).abs() <= 1e-3 * size.max(1.0), "{}: fast {f} vs plain {p}", kind_name(kind));
        }
    }

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
        assert_eq!(out[0], 2.0); // low half of 0x9C = 12 → (12 - 8) × 0.5
        assert_eq!(out[16], 0.5); // high half of 0x9C = 9 → (9 - 8) × 0.5
        assert_eq!(out[1], 0.0); // 8 → (8 - 8) × 0.5
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

    #[test]
    fn fast_matvec_gives_the_same_answer() {
        // A 64 × 64 Q4_0 matrix with varied bytes, scale 0.25 everywhere
        let (rows, cols) = (64, 64);
        let mut bytes = Vec::new();
        for i in 0..rows * cols / 32 {
            bytes.extend_from_slice(&[0x00, 0x34]); // 0x3400 = 0.25
            for j in 0..16 {
                bytes.push(((i * 31 + j * 7) % 256) as u8);
            }
        }
        let m = QMatrix { kind: 2, rows, cols, bytes };
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let simple = m.matvec_simple(&x);
        let fast = m.matvec(&x);
        // The fast version rounds x to 8 bits (integer maths), so allow a tiny difference
        let biggest = simple.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (a, b) in simple.iter().zip(&fast) {
            assert!((a - b).abs() < 0.01 * biggest, "{a} vs {b}");
        }
    }

    #[test]
    fn q8_0_rounding_is_close() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let q = quantize_q8_0(&x);
        assert_eq!(q.scales.len(), 2);
        for (i, v) in x.iter().enumerate() {
            let back = q.values[i] as f32 * q.scales[i / 32];
            assert!((back - v).abs() <= q.scales[i / 32] * 0.5 + 1e-6, "{v} came back as {back}");
        }
        assert_eq!(quantize_q8_0(&[0.0; 32]).values, vec![0i8; 32]); // all zeros stay zeros
    }

    #[test]
    fn integer_dot_matches_decimal_dot() {
        // 4 blocks with every possible 4-bit value and varied scales
        let mut row = Vec::new();
        for b in 0..4u16 {
            row.extend_from_slice(&(0x3000u16 + b * 0x0400).to_le_bytes());
            for j in 0..16u8 {
                row.push(j.wrapping_mul(17).wrapping_add(b as u8 * 3));
            }
        }
        let x: Vec<f32> = (0..128).map(|i| ((i * 37) % 19) as f32 / 7.0 - 1.3).collect();
        let q8 = quantize_q8_0(&x);
        let decimal = dot_q4_0_scalar(&row, &x);
        let plain = dot_q4_0_q8_0_scalar(&row, &q8);
        let fast = dot_q4_0_q8_0(&row, &q8);
        assert!((plain - fast).abs() < 1e-3, "plain {plain} vs avx2 {fast}"); // same integer maths
        assert!((plain - decimal).abs() < 0.02 * decimal.abs().max(1.0), "{plain} vs decimal {decimal}"); // tiny rounding
    }

    #[test]
    fn simd_dot_matches_plain_dot() {
        // 4 blocks (128 numbers) with every possible 4-bit value and varied scales
        let mut row = Vec::new();
        for b in 0..4u16 {
            row.extend_from_slice(&(0x3000u16 + b * 0x0400).to_le_bytes()); // scales 0.125, 0.25, 0.5, 1
            for j in 0..16u8 {
                row.push(j.wrapping_mul(17).wrapping_add(b as u8 * 3));
            }
        }
        let x: Vec<f32> = (0..128).map(|i| ((i * 37) % 19) as f32 / 7.0 - 1.3).collect();
        let plain = dot_q4_0_scalar(&row, &x);
        let fast = dot_q4_0(&row, &x);
        assert!((plain - fast).abs() < 1e-3, "{plain} vs {fast}");
    }

    #[test]
    fn q6_k_dot_matches_unpack_then_dot() {
        // 2 blocks of pseudo-random bytes with a small d, against the plain unpack + dot
        let mut row = Vec::new();
        let mut seed = 7u32;
        for _ in 0..2 {
            for _ in 0..208 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                row.push((seed >> 16) as u8);
            }
            row.extend_from_slice(&0x2400u16.to_le_bytes()); // d = 2^-6
        }
        let x: Vec<f32> = (0..512).map(|i| ((i * 37) % 23) as f32 / 9.0 - 1.1).collect();
        let mut unpacked = Vec::new();
        dequantize_q6_k(&row, &mut unpacked);
        let expected: f32 = unpacked.iter().zip(&x).map(|(a, b)| a * b).sum();
        for got in [dot_q6_k_scalar(&row, &x), dot_q6_k(&row, &x)] {
            assert!((got - expected).abs() < 1e-2 * expected.abs().max(1.0), "{got} vs {expected}");
        }
    }
}