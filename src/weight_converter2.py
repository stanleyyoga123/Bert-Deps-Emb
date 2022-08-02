import numpy as np
import pickle
import os

from tqdm import tqdm

def parse(path):
    f = open(path, "r", encoding="utf-8")
    content = f.read()
    lines = content.split("\n")[:-1]

    word_embeddings = {}

    for line in tqdm(lines):
        splitted = line.split()
        word = splitted[0]
        embedding = np.array(splitted[1:]).astype("double")
        word_embeddings[word] = embedding

    return word_embeddings

embeddings = parse("resources/deps.contexts")
pickle.dump(embeddings, open("resources/embeddings-contextsonly-raw.pkl", "wb"))