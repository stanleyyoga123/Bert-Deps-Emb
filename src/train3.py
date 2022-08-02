import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import os
import argparse

torch.manual_seed(42)

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from model import BertDependencyModel, BertDependencyLocalModel, BertDependencyLocalConcatModel, DependencyLSTMLocalModel, BertClassifier, DependencyLSTMModel
from loader import SSTDepsDataset, CoLADepsDataset, SSTDepsExtDataset
from pytorch_pretrained_bert.optimization import BertAdam
from trainer.default_trainer import default_train
from trainer.deps_trainer import deps_train

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--ww", type=float, default=0.5)
    parser.add_argument("--dw", type=float, default=0.5)
    parser.add_argument("--tokenizer", type=str, default="deps-wordsonly")
    
    args = parser.parse_args()
    ds_type = args.ds
    model_type = args.model
    gpu = args.gpu
    name = args.name
    ww = args.ww
    dw = args.dw
    tokenizer_name = args.tokenizer

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("Configurations")
    print("ds_type: ", ds_type)
    print("model_type: ", model_type)

    tokenizers = [
        BertTokenizer.from_pretrained("bert-base-uncased"),
        BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-{tokenizer_name}")
    ]
    deps_tokenizer = BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-deps-raw")


    if model_type == "bert":
        model = BertClassifier.from_pretrained("bert-base-uncased")
        trainer = default_train

    elif model_type == "bert-dep":
        model = BertDependencyModel.from_pretrained("bert-base-uncased", num_labels=2)
        state_dict = model.create_embeddings(path=f"resources/embeddings/state-dict-{name}.pkl")
        model.add_special_ids(tokenizers[1].vocab["[PAD]"], deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        train = default_train

    elif model_type == "bert-dep-local":
        model = BertDependencyLocalModel.from_pretrained("bert-base-uncased")
        model.create_embeddings("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl")
        model.add_special_ids(tokenizers[1].vocab["[PAD]"], deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        model.add_weights(ww, dw)
        model.add_device(device)
        train = deps_train

    elif model_type == "bert-dep-local-concat":
        model = BertDependencyLocalConcatModel.from_pretrained("bert-base-uncased")
        model.create_embeddings("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl")
        model.add_special_ids(tokenizers[1].vocab["[PAD]"], deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        model.add_device(device)
        train = deps_train

    elif model_type == "bert-dep-ext":
        tokenizers = [
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-{tokenizer_name}")
        ]
        deps_tokenizer = BertTokenizer.from_pretrained("resources/tokenizers/tokenizer-ext-depssonly")
        model = BertDependencyLocalModel.from_pretrained("bert-base-uncased")
        model.create_embeddings("resources/embeddings/state-dict-ext-wordsonly.pkl", "resources/embeddings/state-dict-ext-depssonly.pkl")
        model.add_weights(ww, dw)
        model.add_special_ids(tokenizers[1].vocab["[PAD]"], deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        model.add_device(device)
        train = deps_train
    
    elif model_type == "dep-bilstm":
        model = DependencyLSTMModel(num_labels=2, path="resources/embeddings_state_dict.pkl")
        train = default_train
    
    elif model_type == "dep-avg-bilstm":
        model = DependencyLSTMModel(num_labels=2, path="resources/embeddings_avg_state_dict.pkl")
        train = default_train

    elif model_type == "new-bilstm":
        model = DependencyLSTMModel(num_labels=2, path=f"resources/embeddings/state-dict-{name}.pkl")
        train = default_train
    
    elif model_type == "local-bilstm":
        model = DependencyLSTMLocalModel("resources/embeddings/state-dict-deps-wordsonly.pkl", "resources/embeddings/state-dict-deps-raw.pkl", 2)
        model.add_special_ids(deps_tokenizer.vocab["[UNK]"], deps_tokenizer.vocab["[PAD]"])
        train = deps_train


    if ds_type == "sst" and model_type == "bert-dep-ext":
        print("Use this new loader")
        loader_train = DataLoader(SSTDepsExtDataset("train", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
        loader_dev = DataLoader(SSTDepsExtDataset("dev", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
        loader_test = DataLoader(SSTDepsExtDataset("test", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
    # elif ds_type == "sst":
    #     loader_train = DataLoader(SSTDepsDataset("train", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
    #     loader_dev = DataLoader(SSTDepsDataset("dev", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
    #     loader_test = DataLoader(SSTDepsDataset("test", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
    elif ds_type == "cola":
        loader_train = DataLoader(CoLADepsDataset("train", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
        loader_dev = DataLoader(CoLADepsDataset("dev", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
        loader_test = DataLoader(CoLADepsDataset("test", tokenizers, deps_tokenizer), batch_size=16, shuffle=True)
    else:
        raise Exception("No Type for ds_type")
    model.to(device)
    train(
        model,
        loader_train,
        loader_dev,
        loader_test,
        device,
        name,
        ds_type,
        learning_rate0=args.lr
    )