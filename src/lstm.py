import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# import torchmetrics
# from torchmetrics.classification import MulticlassAccuracy
# from torchmetrics.functional import accuracy
from torchtext.data import Field, TabularDataset, BucketIterator
# import pandas as pd
import nltk
# nltk.download('punkt')
# from sklearn.model_selection import train_test_split
import nltk.data
from nltk.tokenize import word_tokenize
import numpy as np
import pickle

# nltk.data.path.append('src/nltk_data')

def tokenize(text):
    # nltk.data.path.append('./src/nltk_data')
    nltk.data.path.append('./nltk_data')
    return nltk.word_tokenize(text)

# csvファイルを読み込み、データクレンジング
# dataset = pd.read_csv('./dataset.csv')
# dataset = dataset.drop('id', axis=1)
# dataset = dataset.drop('Comments', axis=1)

# 本体制御関連:0 リード関連:1 外部機器関連:2 デバイス構成関連:3 その他:4 ノイズ:5 
# df = dataset['label'].unique()
# dataset = dataset.replace('本体制御関連', 0)
# dataset = dataset.replace('リード関連', 1)
# dataset = dataset.replace('外部機器関連', 2)
# dataset = dataset.replace('デバイス構成関連', 3)
# dataset = dataset.replace('その他', 4)
# dataset = dataset.replace('ノイズ', 5)

# datasetをシャッフル
# dataset = dataset.sample(frac=1.0, random_state=0).reset_index(drop=True)

# datasetをtrain, valに分割
# n_train = int(len(dataset) * 0.7)
# n_val = len(dataset) - n_train
# train, val = train_test_split(dataset, test_size=0.3, stratify=dataset['label'])

# train, valをcsvファイルとして保存
# train.to_csv('./train.csv', index=False, header=False)
# val.to_csv('./val.csv', index=False, header=False)

# TEXT = Field(
#     tokenize=tokenize,
#     lower=True,
#     batch_first=True
# )
# LABEL = Field(
#     sequential=False, 
#     use_vocab = False, 
#     is_target = True
# )

# train, val= TabularDataset.splits(
#     path = 'src/',
#     train = 'train.csv',
#     validation = 'val.csv',
#     format = 'csv',
#     fields =[('x', TEXT), ('t', LABEL)]
# )

# 辞書作成
# TEXT.build_vocab(train, min_freq=3)

# 辞書保存
# vacab_dict = TEXT.vocab.stoi
# with open('vocab_dict.pkl', 'wb') as f:
#     pickle.dump(vacab_dict, f)

# 辞書読み込み
# with open('vocab_dict.pkl', 'rb') as f:
#     vocab_dict = pickle.load(f)

# 辞書サイズ取得
# n_input = len(TEXT.vocab)

# パッディングID取得
# padding_idx = TEXT.vocab.stoi['<pad>']

class LSTMClassifier(pl.LightningModule):

    def __init__(self, n_input=1095, n_output=6, d_model=300, n_hidden=100, num_layers=3, padding_idx=1):
        super().__init__()

        self.embed = nn.Embedding(n_input, d_model, padding_idx)
        self.lstm = nn.LSTM(d_model, n_hidden, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(n_hidden*2, n_output)
        self.n_output = n_output

    def forward(self, x):
        embedding = self.embed(x)
        _, bilstm_hc = self.lstm(embedding)
        bilstm_out = torch.cat((bilstm_hc[0][0], bilstm_hc[0][1]), dim=1)
        output = self.fc(bilstm_out)
        return output

# sentense = 'An implantable medical device having a controlled release biodegradable polymer coating thereon, wherein the polymeric coating comprises: a polymer of the general structure of Formula 6:             wherein a is an integer from 1 to about 20,000; b is an integer from about 1 to about 20,000; c is an integer from about 1 to about 20,000; and the sum of a, b and c is at least 3; and   at least one drug selected from the group consisting of FKBP-12 binding agents, estrogens, chaperone inhibitors, protease inhibitors, protein-tyrosine kinase inhibitors, leptomycin B, peroxisome proliferator-activated receptor gamma ligands (PPARÎ³), hypothemycin, nitric oxide, bisphosphonates, epidermal growth factor inhibitors, antibodies, proteasome inhibitors, antibiotics, anti-inflammatories, anti-sense nucleotides and transforming nucleic acids.'
# text = tokenize(sentense.lower())
# text_idx =[[TEXT.vocab.stoi[x] for x in text]]

# text_idx_ = []
# unk = 0
# for text_ in text:
#     if text_ in vocab_dict.keys():
#         index = vocab_dict[text_]
#         text_idx_.append(index)
#     else:
#         text_idx_.append(unk)
# text_idx_ =[text_idx_]

# text_idx = torch.tensor(text_idx_, dtype=torch.int64)

# net = LSTMClassifier()
# net.load_state_dict(torch.load('src/lstm.pt'))

# net.eval()
    
# # 推論の実行
# with torch.no_grad():
#     y = net(text_idx)
#     y = torch.argmax(F.softmax(y, dim=-1))