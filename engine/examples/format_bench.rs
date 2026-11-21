//! How fast is one matrix × vector for each storage type? (sizes of the 1.5B model's FFN)
//!   cargo run --release --example format_bench
use kinsin_engine::quant::{block_info, kind_name, QMatrix};

fn main() {
    let (rows, cols) = (8960, 1536);
    let x: Vec<f32> = (0..cols).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    for kind in [2u32, 3, 6, 7, 8, 10, 11, 12, 13, 14] {
        let (per_block, block_bytes) = block_info(kind).unwrap();
        let blocks = rows * cols / per_block;
        // Pseudo-random bytes with small, sane f16 scales (0x1000 ≈ 0.00003) in every 2-byte slot
        // we don't know about: good enough for timing, not for maths.
        let mut bytes: Vec<u8> = (0..blocks * block_bytes).map(|i| ((i * 2654435761usize) >> 13) as u8).collect();
        for b in bytes.chunks_exact_mut(block_bytes) {
            for off in [0usize, 2, block_bytes - 4, block_bytes - 2] {
                b[off] = 0x00;
                b[off + 1] = 0x10;
            }
        }
        let m = QMatrix { kind, rows, cols, bytes };
        m.matvec(&x); // warm up
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            std::hint::black_box(m.matvec(&x));
            times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        println!("{:>5}: {:6.2} ms  ({:.1} MB of weights)", kind_name(kind), times[3], m.bytes.len() as f64 / 1e6);
    }
}