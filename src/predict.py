import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import argparse
import pandas as pd

torch.manual_seed(42)

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import BertDependencyLocalModel, DependencyLSTMLocalModel
from loader import SSTDepsDataset, CoLADepsDataset
from pytorch_pretrained_bert.optimization import BertAdam

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm


def predict(model, loader):
    model.eval()
    y_true = np.array([])
    y_preds = np.array([])
    with torch.no_grad():
        for step, batch in enumerate(loader):
            word_ids, deps_ids, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits = model(word_ids, deps_ids)
            y_pred = F.softmax(logits, dim=-1).max(1)[1].cpu().detach().numpy().flatten()
            y_preds = np.concatenate((y_preds, y_pred))
            y_true = np.concatenate((y_true, label.cpu().detach().numpy().flatten()))
    f1 = f1_score(y_true, y_preds, average="weighted")
    acc = accuracy_score(y_true, y_preds)
    print("--------------------------------------------------------------")
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_true, y_preds))
    print()
    print("CLASSIFICATION REPORT")
    print(classification_report(y_true, y_preds))    
    print("--------------------------------------------------------------")
    return f1, acc, y_preds, confusion_matrix(y_true, y_preds), classification_report(y_true, y_preds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="cola")
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    ds = args.ds
    gpu = args.gpu
    model_name = args.model
    path = args.path
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("Configurations")

    word_tokenizers = [BertTokenizer.from_pretrained("bert-base-uncased"), BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-deps-wordsonly")]
    deps_tokenizer = BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-deps-raw")

    if model_name == "bert":
        model = BertDependencyLocalModel.from_pretrained("bert-base-uncased")
        model.create_embeddings("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl")
        model.load_state_dict(torch.load(path)["model_state"])
        model.add_special_ids(deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
    else:
        model = DependencyLSTMLocalModel("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl", 2)
        model.load_state_dict(torch.load(path)["model_state"])
        model.add_special_ids(deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])

    if ds == "cola":
        ds_dev = CoLADepsDataset("dev", word_tokenizers, deps_tokenizer, conv=True)
        ds_test = CoLADepsDataset("test", word_tokenizers, deps_tokenizer, conv=True)
        loader_dev = DataLoader(ds_dev, batch_size=16, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)
    elif ds == "sst":
        ds_dev = SSTDepsDataset("dev", word_tokenizers, deps_tokenizer, conv=True)
        ds_test = SSTDepsDataset("test", word_tokenizers, deps_tokenizer, conv=True)
        loader_dev = DataLoader(ds_dev, batch_size=16, shuffle=False)
        loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    model.to(device)
    f1_dev, acc_dev, y_pred_dev, conf_dev, claf_dev = predict(model, loader_dev)
    f1_test, acc_test, y_pred_test, conf_test, claf_test = predict(model, loader_test)
    best = (conf_dev, claf_dev, conf_test, claf_test, acc_dev, f1_dev, acc_test, f1_test)

    out_dev = ds_dev.data
    out_test = ds_test.data

    out_dev["y_pred"] = y_pred_dev
    out_test["y_pred"] = y_pred_test

    pd.DataFrame(out_dev).to_csv(f"results/dev-{ds}-{model_name}.csv", index=False)
    pd.DataFrame(out_test).to_csv(f"results/test-{ds}-{model_name}.csv", index=False)

    print("Confusion Dev")
    print(best[0])
    print("Classification Report Dev")
    print(best[1])
    print("Confusion Test")
    print(best[2])
    print("Classification Report Test")
    print(best[3])
    print("Acc Dev: ", best[4])
    print("F1 Dev: ", best[5])
    print("Acc Test: ", best[6])
    print("F1 Test: ", best[7])