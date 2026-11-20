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

/// How many bytes one row of `cols` numbers takes in a storage type (None = not supported).
pub fn row_bytes(kind: u32, cols: usize) -> Option<usize> {
    match kind {
        0 => Some(cols * 4),          // F32
        2 => Some(cols / 32 * 18),    // Q4_0
        14 => Some(cols / 256 * 210), // Q6_K
        _ => None,
    }
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
        match self.kind {
            0 => out.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
            2 => dequantize_q4_0(bytes, out),
            14 => dequantize_q6_k(bytes, out),
            _ => unreachable!(),
        }
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

    /// One row · x. Q4_0 rows are multiplied straight from their packed bytes.
    fn row_dot(&self, r: usize, x: &[f32], buf: &mut Vec<f32>) -> f32 {
        if self.kind == 2 {
            let size = row_bytes(2, self.cols).unwrap();
            return dot_q4_0(&self.bytes[r * size..(r + 1) * size], x);
        }
        self.row(r, buf);
        buf.iter().zip(x).map(|(w, v)| w * v).sum()
    }

    /// y = W · x, with the rows shared out between all CPU cores.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols, "vector size doesn't match the matrix");
        let mut y = vec![0.0; self.rows];
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let per_core = self.rows.div_ceil(cores);
        std::thread::scope(|s| {
            // Each core gets its own slice of y to fill: no two cores ever write the same place
            for (i, part) in y.chunks_mut(per_core).enumerate() {
                s.spawn(move || {
                    let first_row = i * per_core;
                    let mut buf = Vec::with_capacity(self.cols);
                    for (j, out) in part.iter_mut().enumerate() {
                        *out = self.row_dot(first_row + j, x, &mut buf);
                    }
                });
            }
        });
        y
    }
}

/// One Q4_0 row · x, straight from the packed bytes (no list of unpacked numbers).
/// Inside a block every number shares one scale, so: add up (value - 8) × x first,
/// then multiply by the scale ONCE per block instead of 32 times.
pub fn dot_q4_0(row: &[u8], x: &[f32]) -> f32 {
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
        for (a, b) in simple.iter().zip(&fast) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}