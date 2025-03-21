use askama::Template;
use learning_lm_rs::model;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

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

// 消息数据结构
#[derive(Clone)]
struct Message {
    role: String,
    content: String,
}

// 主函数
fn main() {
    // 初始化模型和分词器
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let mut messages = Vec::new();
    let mut context_window = Vec::new(); // 维护上下文窗口
    let mut cache = llama.new_cache();
    // let cache = llama.new_cache();

    loop {
        // 用户输入
        print!("User: ");
        let _ = io::stdout().flush();
        let mut input = String::new();
        let _ = io::stdin().read_line(&mut input);
        let input = input.trim();

        if input == "exit" {
            break;
        }

        // 添加用户消息
        messages.push(Message {
            role: "user".to_string(),
            content: input.to_string(),
        });

        // 渲染模板（带生成提示）
        let prompt = ChatPrompt {
            messages: messages.clone(),
            add_generation_prompt: true,
        }
        .render()
        .unwrap();

        // 编码提示词
        let encoding = tokenizer.encode(prompt, true).unwrap();
        let input_ids = encoding.get_ids();

        // 维护上下文窗口（假设模型支持 2048 tokens）
        context_window.extend_from_slice(input_ids);
        if context_window.len() > 2048 {
            context_window.drain(0..context_window.len() - 2048);
            cache.k_cache.drain(0..context_window.len() - 2048);
            cache.v_cache.drain(0..context_window.len() - 2048);
        }

        // 模型生成
        let output_ids = llama.generate(&mut cache, &context_window, 50, 0.8, 30, 1.0);

        // 解码输出
        let response = tokenizer.decode(&output_ids, true).unwrap();
        let response = response.trim();

        // 添加助手回复到上下文
        messages.push(Message {
            role: "assistant".to_string(),
            content: response.to_string(),
        });

        // 更新上下文窗口
        let response_encoding = tokenizer.encode(response, true).unwrap();
        context_window.extend_from_slice(response_encoding.get_ids());

        println!("AI: {}", response);
    }
}
