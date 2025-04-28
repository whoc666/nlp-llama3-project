from transformers import AutoModelForCausalLM, AutoTokenizer

def answer_nlp_question(question, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = "What is tokenization in NLP?"
    print(f"Question: {question}")
    
    answer = answer_nlp_question(question)
    print(f"Answer: {answer}")
