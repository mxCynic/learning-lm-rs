use std::ops::{AddAssign, DivAssign, MulAssign};
use std::vec;
use std::{fs::File, iter::Sum};

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use num_traits::Float;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    _bos_token_id: u32,     // start token id
    eos_token_id: u32,      // end token id
}

impl<T: Copy + Default + Float + MulAssign + Sum + AddAssign + DivAssign + params::FromBytes>
    Llama<T>
{
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params: LLamaParams<T> = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params,
            _bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = q_buf.reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(
                q,
                T::from(0).unwrap(),
                &hidden_states,
                &self.params.wq[layer],
                T::from(1).unwrap(),
            );
            OP::matmul_transb(
                k,
                T::from(0).unwrap(),
                &hidden_states,
                &self.params.wk[layer],
                T::from(1).unwrap(),
            );
            OP::matmul_transb(
                v,
                T::from(0).unwrap(),
                &hidden_states,
                &self.params.wv[layer],
                T::from(1).unwrap(),
            );
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // todo!("self_attention(...)");
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // dbg!(n_groups);
            // dbg!(seq_len);
            // dbg!(total_seq_len);
            // dbg!(self.dqkv);
            // todo!("down_proj matmul and add residual(...)");

            OP::matmul_transb(
                &mut residual,
                T::from(1).unwrap(),
                &hidden_states,
                &self.params.wo[layer],
                T::from(1).unwrap(),
            );

            // todo!("mlp(...)");

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<T>::default(&[1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &[1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &[self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(
            &mut logits,
            T::from(0).unwrap(),
            &hidden_states,
            &self.params.lm_head,
            T::from(1).unwrap(),
        );

        logits
    }

    pub fn generate(
        &self,
        cache: &mut KVCache<T>,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();
        // let mut cache = self.new_cache();
        let input = Tensor::<u32>::new(Vec::from(token_ids), &[token_ids.len()]);
        // 推理用户输入的信息
        let mut tmp = Tensor::<u32>::new(
            vec![OP::random_sample(
                &self.forward(&input, cache),
                top_p,
                top_k,
                temperature,
            )],
            &[1],
        );
        // 获取临时数据

        result.push(tmp.data()[0]);
        for _ in 0..max_len {
            // 更新最新推理的数据
            let word = OP::random_sample(&self.forward(&tmp, cache), top_p, top_k, temperature);
            // dbg!(word);
            // 检查是否为<|end_story|>
            if word == self.eos_token_id {
                break;
            }
            result.push(word);
            unsafe {
                tmp.data_mut()[0] = word;
            }
        }

        result
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention<T: Copy + Default + Float + Sum + AddAssign + DivAssign>(
    hidden_states: &mut Tensor<T>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<T>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // todo!("Implement self_attention");
    let q_data = q.data();
    let k_data = k.data();
    let top_head = n_kv_h * n_groups;
    let alpha = T::from(1. / (dqkv as f32).sqrt()).unwrap();
    let score = unsafe { att_scores.data_mut() };
    let mut score_index = 0;

    score.fill(T::from(0).unwrap());
    // score = Q @ K.T / sqrt(dim)
    for t in 0..top_head {
        for i in 0..seq_len {
            let q_tmp_index = i * top_head * dqkv + t * dqkv;
            let q_tmp = &q_data[q_tmp_index..(q_tmp_index + dqkv)];

            for j in 0..total_seq_len {
                let k_tmp_index = (t / n_groups) * dqkv + j * n_kv_h * dqkv;
                let k_tmp = &k_data[k_tmp_index..(k_tmp_index + dqkv)];

                score[score_index] = k_tmp
                    .iter()
                    .zip(q_tmp.iter())
                    .fold(T::from(0).unwrap(), |acc, (q_val, k_val)| {
                        T::from(acc + *q_val * *k_val).unwrap()
                    })
                    * alpha;
                score_index += 1;
            }
        }
    }
    // attn = softmax(score)
    OP::masked_softmax(att_scores);

    let score = unsafe { att_scores.data_mut() };
    let hidden = unsafe { hidden_states.data_mut() };
    let v_data = v.data();

    hidden.fill(T::from(0).unwrap());
    // attn_V = attn @ V
    for i in 0..top_head {
        let score_len = seq_len * total_seq_len;
        let score_tmp = &score[i * score_len..((i + 1) * score_len)];

        for j in 0..seq_len {
            let tmp_hidden_index = i * dqkv + j * dqkv * top_head;
            let tmp_hidden = &mut hidden[tmp_hidden_index..(tmp_hidden_index + dqkv)];

            for (k, score) in score_tmp[j * total_seq_len..((j + 1) * total_seq_len)]
                .iter()
                .enumerate()
            {
                let tmp_v_index = k * n_kv_h * dqkv + (i / n_groups) * dqkv;
                let tmp_v = &v_data[tmp_v_index..(tmp_v_index + dqkv)];
                tmp_hidden
                    .iter_mut()
                    .zip(tmp_v.iter())
                    .for_each(|(hidden_val, v_val)| *hidden_val += *score * *v_val);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mlp<T: Copy + Default + Float + Sum + AddAssign + MulAssign>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: f32,
) {
    // todo!("Implement mlp");
    // hidden = rms_norm(residual)
    OP::rms_norm(hidden_states, residual, rms_w, eps);

    // gate = hidden @ gate_weight.T
    OP::matmul_transb(
        gate,
        T::from(0).unwrap(),
        hidden_states,
        w_gate,
        T::from(1).unwrap(),
    );

    // up = hidden @ up_weight.T
    OP::matmul_transb(
        up,
        T::from(0).unwrap(),
        hidden_states,
        w_up,
        T::from(1).unwrap(),
    );

    // act = gate * sigmoid(gate) * up ## SwiGLU
    OP::swiglu(up, gate);

    // output = act @ down_weight.T
    OP::matmul_transb(
        residual,
        T::from(1).unwrap(),
        up,
        w_down,
        T::from(1).unwrap(),
    );

    // residual = output + residual
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
