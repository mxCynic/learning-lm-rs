use half::f16;
use learning_lm_rs::model;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    // let model_dir = PathBuf::from(project_dir).join("models").join("story_f16");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f16>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut cache = llama.new_cache();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);

    // let mut cache = llama.new_cache();
    let output_ids = llama.generate(&mut cache, input_ids, 500, 0.8, 30, 1.);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
