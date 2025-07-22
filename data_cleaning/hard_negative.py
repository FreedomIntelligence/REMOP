
from src.contriever import Contriever
from transformers import AutoTokenizer

if __name__ == '__main__':
    model_path = "facebook/contriever"
    contriever = Contriever.from_pretrained(model_path) 
    tokenizer = AutoTokenizer.from_pretrained(model_path)


sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings = contriever(**inputs)

score01 = embeddings[0] @ embeddings[1] #1.0473
score02 = embeddings[0] @ embeddings[2] #1.0095
print(score01, score02)