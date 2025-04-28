from datasets import load_dataset
dataset = load_dataset("squad", split="train[:100]")
print(dataset[0])