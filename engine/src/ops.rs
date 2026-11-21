//! The maths building blocks of a transformer, written by hand.

/// y = W · x
/// W is stored row by row: `rows` rows, each with `x.len()` numbers.
/// Each output number is one row "dotted" with x.
pub fn matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let cols = x.len();
    assert_eq!(w.len(), rows * cols, "matrix size doesn't match");
    let mut y = vec![0.0; rows];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols]; // row r sits at position r × cols
        let mut sum = 0.0;
        for c in 0..cols {
            sum += row[c] * x[c];
        }
        y[r] = sum;
    }
    y
}

/// RMSNorm: rescale x so its "average size" is 1, then multiply by the learned weights.
/// y[i] = x[i] / √(mean(x²) + eps) × weight[i]
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), weight.len());
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
}

/// softmax: turn scores into probabilities (all between 0 and 1, adding up to 1).
/// Changes x in place. Subtracting the biggest score first stops e^x from overflowing.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// SiLU: a smooth "gate". Big positive numbers pass, negative ones are squashed towards 0.
/// silu(x) = x / (1 + e^-x)
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The feed-forward gate (SwiGLU): gate[i] = silu(gate[i]) × up[i], in place.
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    for (g, u) in gate.iter_mut().zip(up) {
        *g = silu(*g) * u;
    }
}

/// RoPE: turn each pair of numbers (i, i + half) by an angle that depends on the word's
/// position, so attention can tell word order. Works on ONE head (e.g. 128 numbers), in place.
pub fn rope(x: &mut [f32], pos: usize, base: f32) {
    let half = x.len() / 2;
    for i in 0..half {
        // Each pair spins at its own speed: fast for the first pairs, slow for the last
        let speed = base.powf(-2.0 * i as f32 / x.len() as f32);
        let angle = pos as f32 * speed;
        let (sin, cos) = angle.sin_cos();
        let a = x[i];
        let b = x[i + half];
        x[i] = a * cos - b * sin; // turn the arrow (a, b) by `angle`
        x[i + half] = a * sin + b * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        #[test]
    fn rope_by_hand() {
        // Position 0: no turn at all
        let mut x = [1.0, 0.0, 0.0, 0.0];
        rope(&mut x, 0, 10_000.0);
        assert_eq!(x, [1.0, 0.0, 0.0, 0.0]);

        // Position 1, base 1 → every pair turns by exactly 1 radian
        let mut x = [1.0, 0.0, 0.0, 0.0];
        rope(&mut x, 1, 1.0);
        assert!((x[0] - 0.5403).abs() < 1e-4); // cos(1)
        assert!((x[2] - 0.8415).abs() < 1e-4); // sin(1)
    }

    #[test]
    fn rope_only_the_distance_matters() {
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let at = |v: &[f32], pos: usize| {
            let mut v = v.to_vec();
            rope(&mut v, pos, 10_000.0);
            v
        };
        let q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let k = [0.8, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, 0.9];
        let two_apart = dot(&at(&q, 5), &at(&k, 3)); // positions 5 and 3
        let also_two_apart = dot(&at(&q, 7), &at(&k, 5)); // positions 7 and 5
        let six_apart = dot(&at(&q, 9), &at(&k, 3)); // positions 9 and 3
        assert!((two_apart - also_two_apart).abs() < 1e-4); // same distance → same score
        assert!((two_apart - six_apart).abs() > 0.1); // different distance → different score
    }
    #[test]
    fn silu_by_hand() {
        assert_eq!(silu(0.0), 0.0);
        assert!((silu(1.0) - 0.7311).abs() < 1e-4);   // 1 / (1 + e^-1)
        assert!((silu(-1.0) + 0.2689).abs() < 1e-4);  // squashed towards 0
        assert!((silu(10.0) - 10.0).abs() < 1e-3);    // big numbers pass almost unchanged
    }

    #[test]
    fn swiglu_by_hand() {
        let mut gate = [0.0, 1.0];
        swiglu(&mut gate, &[5.0, 2.0]);
        assert_eq!(gate[0], 0.0);                     // silu(0) × 5 = 0: gate closed
        assert!((gate[1] - 1.4622).abs() < 1e-4);     // silu(1) × 2
    }
    #[test]
    fn softmax_by_hand() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x[0] - 0.0900).abs() < 1e-4);
        assert!((x[1] - 0.2447).abs() < 1e-4);
        assert!((x[2] - 0.6652).abs() < 1e-4);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6); // adds up to 1
    }

    #[test]
    fn softmax_huge_scores_dont_break() {
        let mut x = [1000.0, 1001.0]; // e^1000 alone would be infinity
        softmax(&mut x);
        assert!((x[0] - 0.2689).abs() < 1e-4);
        assert!((x[1] - 0.7311).abs() < 1e-4);
    }
    #[test]
    fn rms_norm_by_hand() {
        let y = rms_norm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
        assert!((y[0] - 0.84853).abs() < 1e-4); // 3 / √12.5
        assert!((y[1] - 2.26274).abs() < 1e-4); // 4 / √12.5 × 2
    }


    #[test]
    fn matvec_by_hand() {
        // W = | 1 2 3 |    x = | 1 |    W·x = | 1·1 + 2·0 + 3·2 |  = |  7 |
        //     | 4 5 6 |        | 0 |          | 4·1 + 5·0 + 6·2 |    | 16 |
        //                      | 2 |
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, 2.0];
        assert_eq!(matvec(&w, &x, 2), vec![7.0, 16.0]);
    }
}