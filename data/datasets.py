import pandas as pd
import os
import torch
import json
from sklearn.model_selection import train_test_split
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


def get_compiled_dataset(path, allowed_labelers=['Vasily', '']):
    def read_csv_dataset(path):
        df = pd.read_csv(path, sep=',')
        df = df[['Requirement', 'Context (Keywords)', 'Name of Doc', 'Label', 'Comments.1', 'Labeled by.1']]
        df.columns = ['text', 'context', 'doc', 'label', 'comments', 'labeler']

        # clean data from mess for timebeing

        df = df.drop(df[(df['label'] != 'Access control') & (df['label'] != 'Confidentiality') \
                        & (df['label'] != 'Availability') \
                        & (df['label'] != 'Integrity') & (df['label'] != 'Operational') & \
                        (df['label'] != 'Accountability')].index)

        # clean from dirty labelers

        df = df[~df['label'].isin(allowed_labelers)]

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

    train, test = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = OwnDataset(train, tokenizer)
    test_dataset = OwnDataset(test, tokenizer)

    return train_dataset, test_dataset, encode_dict
