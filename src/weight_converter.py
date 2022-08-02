import numpy as np
import pickle
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
import argparse
import os
import torch

def main(path, names, maxlen, is_zero):
    VOCAB_PATH = f"resources/tokenizers/vocab-{names}.txt"
    TOKENIZER_PATH = f"resources/tokenizers/tokenizer-{names}"
    print("Loading Embeddings")
    embeddings = pickle.load(open(path, "rb"))
    special_tokens = ["[UNK]", "[SEP]", "[PAD]", "[CLS]"]
    words = list(embeddings.keys()) + special_tokens

    print("Creating Vocabs")
    with open(os.path.join(VOCAB_PATH), "w", encoding="utf-8") as f:
        for word in words:
            f.write(f"{word}\n")
    
    print("Creating Tokenizer")
    tokenizer = BertTokenizer(VOCAB_PATH, model_max_length=maxlen, maxlen=maxlen)
    tokenizer.save_pretrained(TOKENIZER_PATH)

    print("Creating Dummy Bert Model")
    cfg = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=300)
    bert = BertModel(cfg)

    new_embeddings = [[] for _ in range(len(tokenizer.vocab))]

    for word in tqdm(tokenizer.vocab):
        idx = tokenizer.vocab[word]
        if word not in embeddings:
            new_embeddings[idx] = bert.embeddings.word_embeddings.weight[idx].detach().numpy()
            continue
        new_embeddings[idx] = np.array(embeddings[word], dtype=float)
        if len(new_embeddings[idx] != maxlen):
            new_embeddings[idx] = new_embeddings[idx][len(new_embeddings[idx]) - maxlen:]

    for i, el in enumerate(new_embeddings):
        if el.dtype != new_embeddings[0].dtype:
            new_embeddings[i] = new_embeddings[i].astype("float64")
    
    if is_zero:
        for token in special_tokens:
            idx = tokenizer.vocab[token]
            new_embeddings[idx] = np.array([0 for _ in range(300)])
        

    print("Creating State Dict")
    new_state_dict = torch.from_numpy(np.array(new_embeddings))
    pickle.dump(new_state_dict, open(f"resources/embeddings/state-dict-{names}.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--names", type=str, required=True)
    parser.add_argument("--maxlen", type=int, required=True)
    parser.add_argument("--zero", type=int, required=True)
    
    args = parser.parse_args()
    path = args.path
    names = args.names
    maxlen = args.maxlen
    is_zero = args.zero

    main(path, names, maxlen, is_zero)