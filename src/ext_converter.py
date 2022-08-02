import numpy as np
import pickle
import os

from tqdm import tqdm

def parse(content):
    lines = content.split("\n")[:-1]

    word_embeddings = {}
    errors = []
    for line in tqdm(lines):
        splitted = line.split()
        word = splitted[0]
        
        try:
            if "/" in splitted[1]:
                splitted[1] = eval(splitted[1])
            embedding = np.array(splitted[1:]).astype("double")
        
        except Exception as e:
            errors.append((splitted, e))
        word_embeddings[word] = embedding
    return word_embeddings

path = "resources/wiki_extvec"
f = open(path, "r", encoding="utf-8")
content = f.read()

word_embeddings_ = parse(content)
words_ = list(word_embeddings_.keys())
contexts = [word.split("_") for word in tqdm(words_)]
contexts = [el for el in contexts if len(el) > 1]

words = []
for el in words_:
    if len(el.split("_")) == 1:
        words.append(el)
        
contexts = ["_".join(el) for el in contexts]
word_embeddings = { word: word_embeddings_[word] for word in words }
deps_embeddings = { word: word_embeddings_[word] for word in contexts }

pickle.dump(word_embeddings, open("resources/embeddings-wordsonly-ext.pkl", "wb"))
pickle.dump(deps_embeddings, open("resources/embeddings-depssonly-ext.pkl", "wb"))