use askama::Template;

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
)] // 禁用 HTML 转义
pub struct ChatTemplate<'a> {
    pub messages: &'a [Message],
    pub add_generation_prompt: bool,
}

#[derive(serde::Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: &'static str,
}
