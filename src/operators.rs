// use num_traits::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign};

use num_traits::Float;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather<T: Copy + Default>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope<T: Copy + Default + Float>(y: &mut Tensor<T>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = T::from(pos).unwrap()
                    / T::from(theta)
                        .unwrap()
                        .powf(T::from(i * 2).unwrap() / T::from(d).unwrap());
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<T: Copy + Default + DivAssign + Sum + Float>(y: &mut Tensor<T>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<T>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::from(0).unwrap());
        }
    }
}

pub fn rms_norm<T: Copy + Default + Float + Sum>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
    w: &Tensor<T>,
    epsilon: f32,
) {
    // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
    let data_x = x.data();
    let data_w = w.data();
    let mdata_y = unsafe { y.data_mut() };
    let strides = w.shape()[0];

    let square_x_norms: Vec<T> = data_x
        .chunks(strides)
        .flat_map(|chunks| {
            let sum: T = chunks.iter().map(|x| x.powi(2)).sum();
            vec![sum / T::from(strides).unwrap(); strides]
        })
        .collect();

    for (i, y2) in mdata_y.iter_mut().enumerate() {
        *y2 = (data_x[i] * data_w[i % strides])
            / (square_x_norms[i] + T::from(epsilon).unwrap()).sqrt()
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu<T: Copy + Default + Float + MulAssign>(y: &mut Tensor<T>, x: &Tensor<T>) {
    // let len = y.size();
    // assert!(len == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    let y1 = unsafe { y.data_mut() };

    for (i, y_) in y1.iter_mut().enumerate() {
        *y_ *= (T::from(1).unwrap() / (T::from(1).unwrap() + T::exp(-x.data()[i]))) * x.data()[i];
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T: Copy + Default + Float + Sum + MulAssign + AddAssign>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: T,
) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");

    let data_a = a.data();
    let data_b = b.data();
    let data_c = unsafe { c.data_mut() };

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    let mut a_times_b: Vec<T> = Vec::new();

    for i in 0..m {
        for j in 0..n {
            let element: T = (0..k).map(|x| data_a[i * k + x] * data_b[j * k + x]).sum();
            a_times_b.push(element);
        }
    }

    for (i, c_ele) in data_c.iter_mut().enumerate() {
        *c_ele *= beta;
        *c_ele += alpha * a_times_b[i];
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T: Copy + Default + Float + AddAssign>(x: &Tensor<T>, y: &Tensor<T>) -> T {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = T::from(0).unwrap();
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T: Copy + Default + Float>(
    x: &Tensor<T>,
    top_p: f32,
    top_k: u32,
    temperature: f32,
) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability<T> {
        val: T,
        tok: u32,
    }
    impl<T: Float> Eq for Probability<T> {}
    impl<T: Float + PartialOrd> PartialOrd for Probability<T> {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<T: Float> Ord for Probability<T> {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.partial_cmp(&other.val).unwrap_or(Ordering::Less) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl<T: Float + Clone> From<(usize, &T)> for Probability<T> {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: *p,
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, T::from(1).unwrap());
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val =
            logits[i - 1].val + ((logits[i].val - max) / T::from(temperature).unwrap()).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * T::from(top_p).unwrap();
    let plimit = T::from(rand::random::<f32>()).unwrap() * T::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
