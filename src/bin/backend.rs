use axum::http::Method;
use axum::{routing::post, Json, Router};
use axum_server::Server;
use learning_lm_rs::kvcache::KVCache;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Mutex, OnceLock};
use tower_http::cors::CorsLayer;

use askama::Template;
use learning_lm_rs::model;
use once_cell::sync::OnceCell;
use std::path::PathBuf;
use tokenizers::Tokenizer;

static MODEL: OnceCell<model::Llama<f32>> = OnceCell::new();
// static CACHE: OnceCell<KVCache<f32>> = OnceCell::new();
static CACHE: OnceLock<Mutex<KVCache<f32>>> = OnceLock::new();
static TOKENIZER: OnceCell<Tokenizer> = OnceCell::new();

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
}

#[derive(Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}
#[derive(Deserialize)]
struct InputText {
    text: String,
}

#[derive(Serialize)]
struct OutputText {
    response: String,
}

// 定义模板结构
#[derive(Template)]
#[template(
    source = r#"
{% for message in messages %}
{{ "<|im_start|>" }}{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}
{{ "<|im_start|>assistant\n" }}
{% endif %}
"#,
    ext = "txt",
    escape = "none"
)]
struct ChatPrompt {
    messages: Vec<Message>,
    add_generation_prompt: bool,
}
async fn story_text(Json(input): Json<InputText>) -> Json<OutputText> {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let binding = tokenizer.encode(input.text.clone(), true).unwrap();
    let input_ids = binding.get_ids();
    let mut cache = llama.new_cache();
    let output_ids = llama.generate(&mut cache, input_ids, 500, 0.9, 30, 1.);
    let output = tokenizer.decode(&output_ids, true).unwrap();
    Json(OutputText {
        response: format!("{}{}", input.text, output),
    })
}

async fn chat(Json(payload): Json<ChatRequest>) -> Json<OutputText> {
    // 初始化模型和分词器（首次请求时加载）
    let llama = MODEL.get_or_init(|| {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        model::Llama::<f32>::from_safetensors(&model_dir)
    });

    let tokenizer = TOKENIZER.get_or_init(|| {
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap()
    });
    let cache = CACHE.get_or_init(|| Mutex::new(llama.new_cache()));

    let mut cache_data = cache.lock().unwrap();
    // 构建提示模板
    let prompt = ChatPrompt {
        messages: payload.messages,
        add_generation_prompt: true,
    }
    .render()
    .unwrap();

    // 编码输入
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let input_ids = encoding.get_ids();

    // 生成回复
    let output_ids = llama.generate(&mut (*cache_data), input_ids, 500, 0.8, 30, 1.0);
    let response = tokenizer.decode(&output_ids, true).unwrap();
    let cleaned_response = response
        .trim()
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "");

    Json(OutputText {
        response: cleaned_response,
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/api/Story", post(story_text))
        .route("/api/Chat", post(chat))
        .layer(
            CorsLayer::new()
                .allow_origin(["http://localhost:4321".parse().unwrap()])
                .allow_methods([Method::POST, Method::OPTIONS])
                .allow_headers([axum::http::header::CONTENT_TYPE]),
        );

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("服务器运行在 http://{}", addr);
    Server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
