//! Golden test: every storage type must unpack to exactly what the reference
//! implementation (gguf-py, the Python library from llama.cpp) gives.
//! The data file was made by scripts/make_golden_quants.py.

use kinsin_engine::quant::{block_info, dequantize, kind_name};

fn u32_at(data: &[u8], at: &mut usize) -> u32 {
    let v = u32::from_le_bytes(data[*at..*at + 4].try_into().unwrap());
    *at += 4;
    v
}

#[test]
fn every_format_matches_the_reference() {
    let data = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/golden_quants.bin")).unwrap();
    let mut at = 0;
    let mut checked = Vec::new();
    while at < data.len() {
        let kind = u32_at(&data, &mut at);
        let len = u32_at(&data, &mut at) as usize;
        let packed = &data[at..at + len];
        at += len;
        let count = u32_at(&data, &mut at) as usize;
        let expected: Vec<f32> = data[at..at + count * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        at += count * 4;

        assert!(block_info(kind).is_some(), "{} should be supported", kind_name(kind));
        let mut got = Vec::new();
        dequantize(kind, packed, &mut got);
        assert_eq!(got.len(), expected.len(), "{}: wrong count", kind_name(kind));
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            // Same maths, maybe in a different order: allow a tiny rounding difference
            let tolerance = 1e-5 * e.abs().max(1.0);
            assert!((g - e).abs() <= tolerance, "{} number {i}: got {g}, reference says {e}", kind_name(kind));
        }
        checked.push(kind_name(kind));
    }
    assert_eq!(checked.len(), 12, "checked: {checked:?}");
}