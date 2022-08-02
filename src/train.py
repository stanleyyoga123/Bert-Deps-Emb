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

from model import BertDependencyModel, BertClassifier, DependencyLSTMModel
from loader import SSTDataset, CoLADataset
from pytorch_pretrained_bert.optimization import BertAdam

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm


def predict(model, loader):
    model.eval()
    y_true = np.array([])
    y_preds = np.array([])
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids, label = batch[0].to(device), batch[1].to(device)
            logits = model(input_ids)
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
    parser.add_argument("--ds", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)
    
    args = parser.parse_args()
    ds_type = args.ds
    model_type = args.model
    gpu = args.gpu
    name = args.name

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    print("Configurations")
    print("ds_type: ", ds_type)
    print("model_type: ", model_type)

    if model_type == "bert":
        tokenizers = [BertTokenizer.from_pretrained("bert-base-uncased")]
        model = BertClassifier.from_pretrained("bert-base-uncased")

    elif model_type == "bert-dep":
        tokenizers = [
            BertTokenizer.from_pretrained("bert-base-uncased"),
            BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-{name}")
        ]
        model = BertDependencyModel.from_pretrained("bert-base-uncased", num_labels=2)
        state_dict = model.create_embeddings(path=f"resources/embeddings/state-dict-{name}.pkl")
        print(state_dict.shape)
    
    elif model_type == "dep-bilstm":
        tokenizers = [BertTokenizer.from_pretrained("resources/tokenizer-bert-dependency-context")]
        model = DependencyLSTMModel(num_labels=2, path="resources/embeddings_state_dict.pkl")
    
    elif model_type == "dep-avg-bilstm":
        tokenizers = [BertTokenizer.from_pretrained("resources/tokenizer-bert-dependency-context")]
        model = DependencyLSTMModel(num_labels=2, path="resources/embeddings_avg_state_dict.pkl")

    elif model_type == "new-bilstm":
        tokenizers = [BertTokenizer.from_pretrained(f"resources/tokenizers/tokenizer-{name}")]
        model = DependencyLSTMModel(num_labels=2, path=f"resources/embeddings/state-dict-{name}.pkl")

    if ds_type == "sst":
        loader_train = DataLoader(SSTDataset("train", tokenizers), batch_size=16, shuffle=True)
        loader_dev = DataLoader(SSTDataset("dev", tokenizers), batch_size=16, shuffle=True)
        loader_test = DataLoader(SSTDataset("test", tokenizers), batch_size=16, shuffle=True)
    elif ds_type == "cola":
        loader_train = DataLoader(CoLADataset("train", tokenizers), batch_size=16, shuffle=True)
        loader_dev = DataLoader(CoLADataset("dev", tokenizers), batch_size=16, shuffle=True)
        loader_test = DataLoader(CoLADataset("test", tokenizers), batch_size=16, shuffle=True)
    else:
        raise Exception("No Type for ds_type")

    model.to(device)

    epochs = 20
    warmup_proportion = 0.1
    gradient_accumulation_steps = 1
    learning_rate0 = args.lr
    l2_decay = 0.01
    num_classes = 2

    print("Configs")
    print(f"Epochs: {epochs}")
    print(f"learning_rate: {learning_rate0}")

    total_train_steps = int(len(loader_train) / gradient_accumulation_steps * epochs)

    optimizer = BertAdam(
        model.parameters(),
        lr=learning_rate0,
        warmup=warmup_proportion,
        t_total=total_train_steps,
        weight_decay=l2_decay,
    )

    best_dev = 0
    best_test = 0
    for epoch in range(epochs):
        tr_loss = 0
        ep_train_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        for step, batch in enumerate(loader_train):
            input_ids, label = batch[0].to(device), batch[1].to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, num_classes), label)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            
            tr_loss += loss
            optimizer.step()
            optimizer.zero_grad()

            if step % 40 == 0:
                print(
                    "Epoch:{}-{}/{}, Train {} Loss: {} ".format(
                        epoch+1,
                        step,
                        len(loader_train),
                        "Cross Entropy",
                        loss.item(),
                    )
                )
        f1_dev, acc_dev, _, conf_dev, claf_dev = predict(model, loader_dev)
        f1_test, acc_test, _, conf_test, claf_test = predict(model, loader_test)
        print("-----------------------------------------")
        print("F1 Dev: ", f1_dev)
        print("Acc Dev: ", acc_dev)
        print("F1 Test: ", f1_test)
        print("Acc Test: ", acc_test)
        print("-----------------------------------------")

        if f1_test > best_test:
            to_save = {
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
            torch.save(to_save, os.path.join("save", f"{model_type}-{ds_type}-{name}-model-best-test-{epoch}"))
            best_test = f1_test

        if f1_dev > best_dev:
            to_save = {
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
            torch.save(to_save, os.path.join("save", f"{model_type}-{ds_type}-{name}-model-best-dev-{epoch}"))
            best_dev = f1_dev
            best = (conf_dev, claf_dev, conf_test, claf_test, acc_dev, f1_dev, acc_test, f1_test)
    
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
