import pandas as pd
import os
import torch
import json
from sklearn.model_selection import train_test_split, KFold
from transformers import DistilBertTokenizerFast


def get_secreq_dataset(path):
    def read_dataset(path):
        texts, labels = [], []
        for f in os.listdir(path):
            filepath = os.path.join(path, f)
            df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'is_sec'])
            df.dropna(inplace=True)
            texts.extend([text.strip() for text in df['text'].to_list()])
            labels.extend([1 if label == 'sec' else 0 for label in df['is_sec'].to_list()])
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test

    train_texts, test_texts, train_labels, test_labels = read_dataset(path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class SecReqDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]).to(device)
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SecReqDataset(train_encodings, train_labels)
    test_dataset = SecReqDataset(test_encodings, test_labels)

    return train_dataset, test_dataset


def get_riaz_dataset(path):
    def read_json_dataset(path):
        texts, labels = [], []
        listdir = os.listdir(path)
        for f in listdir:
            print(f)
            filepath = os.path.join(path, f)
            with open(filepath, encoding='utf-8') as f:
                jj = json.load(f)
                jj = jj['content']
                for item in jj:
                    if item['securityObjectiveAnnotationsDefined'] == True:
                        objective = ""
                        objective_count = 0
                        for itemm in item["securityObjectiveAnnotations"]:
                            if itemm["securityImpact"] == 'MODERATE' or itemm["securityImpact"] == 'HIGH':
                                if objective_count > 0:
                                    objective += ','
                                objective += itemm["securityObjective"]
                                objective_count += 1
                        if objective != "":
                            texts.append(item['sentence'])
                            labels.append(objective)

        encode_dict = {}

        def encode_cat(x):
            if x not in encode_dict.keys():
                encode_dict[x] = len(encode_dict)
            return encode_dict[x]

        encoded_labels = [encode_cat(label) for label in labels]

        # d = {'text': texts, 'document': origin, 'label': labels, 'encoded_label': encoded_labels}
        # df = pd.DataFrame(data=d)

        # return df

        X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test, encode_dict

    # riaz_df = read_json_dataset('Riaz-Dataset-main')
    # X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.1, random_state=42)

    train_texts, test_texts, train_labels, test_labels, encode_dict = read_json_dataset('Riaz-Dataset-main')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class RiazDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]).to(device)
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = RiazDataset(train_encodings, train_labels)
    test_dataset = RiazDataset(test_encodings, test_labels)
    return train_dataset, test_dataset, encode_dict


def get_compiled_dataset(path: str, type: str = "split", test_size: float = 0.1,
                         allowed_labelers: list = ['Vasily', 'Aydar'],
                         allowed_labels: list = ['Access control', 'Confidentiality', 'Availability', 'Integrity',
                                                 'Operational', 'Accountability']):
    """

    :param path: path to dataset file
    :param type (optional): either 'split' or '10-fold': 'split' will split data to train and test datasets, '10-fold' will split data into 10 train/test splits.
    :param test_size (optional): how big test dataset is, used only if type is 'split'
    :param allowed_labelers (optional): samples from which labelers are allowed
    :param allowed_labels (optional): samples with which labels are allowed
    :return: This function returns three objects:
        - train – train dataset or list of train datasets if ``type`` was set to 'split'
        - test – test dataset or list of test datasets if `type` was set to 'split'
        - encode_dict – dictionary which maps labels' names to labels' ids
    """

    def read_csv_dataset(path):
        df = pd.read_csv(path, sep=',')
        df = df[['Requirement', 'Context (Keywords)', 'Name of Doc', 'Label', 'Comments.1', 'Labeled by.1']]
        df.columns = ['text', 'context', 'doc', 'label', 'comments', 'labeler']

        # filter labels
        df = df[(df['label'].isin(allowed_labels))]

        # filter labelers
        df = df[(df['labeler'].isin(allowed_labelers))]

        encode_dict = {}

        def encode_cat(x):
            if x not in encode_dict.keys():
                encode_dict[x] = len(encode_dict)
            return encode_dict[x]

        encoded_labels = [encode_cat(label) for label in df['label'].values]
        df = df.assign(encoded_label=pd.Series(encoded_labels, index=df.index))

        return df, encode_dict

    df, encode_dict = read_csv_dataset(path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    class OwnDataset(torch.utils.data.Dataset):
        def __init__(self, df, encoder):
            self.df = df
            self.encoder = encoder
            self.encodings = tokenizer(list(self.df['text'].values), truncation=True, padding=True)
            self.labels = df['encoded_label'].values

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx]).to(device)
            return item

        def __len__(self):
            return len(self.labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if type == "split":
        train, test = train_test_split(df, test_size=test_size, random_state=42)
        train_dataset = OwnDataset(train, tokenizer)
        test_dataset = OwnDataset(test, tokenizer)
    elif type == "10-fold":
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        train_dataset, test_dataset = [], []
        for train_index, val_index in kfold.split(df):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]
            train_dataset.append(OwnDataset(train_df, tokenizer))
            test_dataset.append(OwnDataset(val_df, tokenizer))
    else:
        raise ValueError("type is either 'split' or '10-fold'")

    return train_dataset, test_dataset, encode_dict
