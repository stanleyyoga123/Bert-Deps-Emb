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

    
embeddings = parse("resources/bow2.words")
pickle.dump(embeddings, open("resources/embeddings-wordsonly-bow2.pkl", "wb"))
embeddings = parse("resources/bow2.contexts")
pickle.dump(embeddings, open("resources/embeddings-contextsonly-bow2.pkl", "wb"))
embeddings = parse("resources/bow5.words")
pickle.dump(embeddings, open("resources/embeddings-wordsonly-bow5.pkl", "wb"))
embeddings = parse("resources/bow5.contexts")
pickle.dump(embeddings, open("resources/embeddings-contextsonly-bow5.pkl", "wb"))
embeddings = parse("resources/deps.words")
pickle.dump(embeddings, open("resources/embeddings-wordsonly-deps.pkl", "wb"))

def parse_2(path):
    f = open(path, "r", encoding="utf-8")
    content = f.read()
    lines = content.split("\n")[:-1]

    word_embeddings = {}

    for line in tqdm(lines):
        splitted = line.split()
        word = "_".join(splitted[0].split("_")[1:])
        embedding = np.array(splitted[1:]).astype("double")

        if word not in word_embeddings:
            word_embeddings[word] = []
        word_embeddings[word].append(embedding)
    
    for key in word_embeddings.keys():
        word_embeddings[key] = np.mean(word_embeddings[key], axis=0)

    return word_embeddings

embeddings = parse_2("resources/deps.contexts")
pickle.dump(embeddings, open("resources/embeddings-contextsonly-deps.pkl", "wb"))

def combine(path1, path2):
    new_embeddings = {}
    embedding1, embedding2 = pickle.load(open(path1, "rb")), pickle.load(open(path2, "rb"))
    words = list(embedding1.keys()) + list(embedding2.keys())
    for word in words:
        embeddings = []
        if word in embedding1:
            embeddings.append(embedding1[word])
        
        if word in embedding2:
            embeddings.append(embedding2[word])
        
        new_embeddings[word] = np.mean(embeddings, axis=0)
    return new_embeddings

pairs = (
    ("resources/embeddings-contextsonly-bow2.pkl", "resources/embeddings-wordsonly-bow2.pkl", "resources/embeddings-bow2-combine.pkl"),
    ("resources/embeddings-contextsonly-bow5.pkl", "resources/embeddings-wordsonly-bow5.pkl", "resources/embeddings-bow5-combine.pkl"),
    ("resources/embeddings-contextsonly-deps.pkl", "resources/embeddings-wordsonly-deps.pkl", "resources/embeddings-deps-combine.pkl"),
)
for path1, path2, target in tqdm(pairs):
    embeddings = combine(path1, path2)
    pickle.dump(embeddings, open(target, "wb"))