from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def predict(model, dataset, encode_dict):
    model.eval()
    y_true, y_pred = [], []

    for i in range(len(dataset)):
        inputs = dataset[i]
        a, b, c = inputs['input_ids'], inputs['attention_mask'], inputs['labels']
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        c = c.unsqueeze(0)
        model_predictions = model(a, attention_mask=b, labels=c)
        model_predictions = model_predictions.logits.argmax(dim=1)
        y_true.append(c.item())
        y_pred.append(model_predictions.item())

    decode_dict = {value: key for key, value in encode_dict.items()}
    y_true = [decode_dict[y] for y in y_true]
    y_pred = [decode_dict[y] for y in y_pred]
    return y_true, y_pred

def show_confusion_matrix(y_true, y_pred, encode_dict):
    categories = [key for key in encode_dict.keys()]
    y_true = pd.Categorical(y_true, categories=categories)
    y_pred = pd.Categorical(y_pred, categories=categories)
    confusion_matrix = pd.crosstab(y_true, y_pred, dropna=False, rownames=['Actual'], colnames=['Predicted'])

    fig, ax = plt.subplots(figsize=(10, 10))
    sn.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.show()
    return None
