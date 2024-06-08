use std::f32::consts::PI;

const TRIANGLE_INDICES: [u16; 3] = [0, 1, 2];
const SQUARE_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

pub fn indices(n: usize) -> Vec<u16> {
    match n {
        0..=2 => panic!("Invalid vertex count"),
        3 => TRIANGLE_INDICES.to_vec(),
        4 => SQUARE_INDICES.to_vec(),
        _ => {
            let mut res = Vec::new();

            for i in (2..=n).step_by(2) {
                for j in i - 2..i {
                    res.push(j as _);
                }
                if i == n {
                    res.push(0);
                } else {
                    res.push(i as _);
                }
            }

            res.extend(indices(n.div_ceil(2)).iter().map(|&i| i * 2));

            res
        }
    }
}

pub fn vertices(n: usize) -> Vec<(f32, f32)> {
    let step = (2. * PI) / n as f32;

    (0..n).map(|i| (step * i as f32).sin_cos()).collect()
}
