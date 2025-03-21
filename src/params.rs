use half::{bf16, f16};
use std::convert::TryInto;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

pub trait FromBytes: Sized {
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        assert!(
            bytes.len() % 4 == 0,
            "Input byte slice must be a multiple of 4"
        );
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

impl FromBytes for f16 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        assert!(
            bytes.len() % 2 == 0,
            "Input byte slice must be a multiple of 2"
        );
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

impl FromBytes for bf16 {
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        assert!(
            bytes.len() % 2 == 0,
            "Input byte slice must be a multiple of 2"
        );
        // let a: Vec<f32> = bytes
        //     .chunks_exact(4)
        //     .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        //     .collect();
        //
        // a.iter().map(|x| bf16::from_f32(*x)).collect()

        bytes
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
}

impl<T: Copy + Default + Clone + FromBytes> LLamaParams<T> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...
        // };

        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        let layers = config.num_hidden_layers;

        // for i in safetensor.names() {
        //     println!("{}", i);
        // }
        let get_tensor = |name: &str| match safetensor.tensor(name) {
            Ok(tensor) => {
                let shape = tensor.shape().to_vec();
                let _data = tensor.data();
                let data: Vec<T> = FromBytes::from_bytes(tensor.data());

                Tensor::new(data, &shape)
            }
            Err(_) => {
                println!("no this tensor: {}", name);
                Tensor::default(&Vec::new())
            }
        };

        let embedding_table = if config.tie_word_embeddings {
            get_tensor("lm_head.weight")
        } else {
            get_tensor("model.embed_tokens.weight")
        };

        // println!("embedding_table");
        // embedding_table.print();
        LLamaParams {
            // embedding_table: get_tensor("lm_head.weight"),
            embedding_table,

            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),

            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
