from transformers import AutoModelForCausalLM, AutoTokenizer

def answer_nlp_question(question, model_name="meta-llama/Llama-3-8b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = "What is tokenization in NLP?"
    print(f"Question: {question}")
    # 推理在 Colab 运行