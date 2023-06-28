from getpatelem import getclaim1
from flask import Flask, request, render_template, redirect
# from torchtext.data import Field, TabularDataset, BucketIterator
from lstm import tokenize, LSTMClassifier
import torch
import torch.nn.functional as F
import pickle

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    if request.method == 'POST':
        number = request.form['patentnumber']
        if number == '':
            return redirect(request.url)
        claim_1st = getclaim1(number)
        if claim_1st == 'not extracted':
            return render_template('result1.html', claim_1st = claim_1st, number = number)
        
        # 形態素解析
        text = tokenize(claim_1st.lower())
        # 辞書読み込み
        # with open('./src/vocab_dict.pkl', 'rb') as f:
        with open('./vocab_dict.pkl', 'rb') as f:
            vocab_dict = pickle.load(f)
        # ID変換
        text_idx = []
        unk = 0
        for text_ in text:
            if text_ in vocab_dict.keys():
                index = vocab_dict[text_]
                text_idx.append(index)
            else:
                text_idx.append(unk)
        text_idx =[text_idx]
        text_idx = torch.tensor(text_idx, dtype=torch.int64)

        net = LSTMClassifier()
        # net.load_state_dict(torch.load('./src/lstm.pt'))
        net.load_state_dict(torch.load('./lstm.pt'))

        net.eval()
            
        # 推論の実行
        with torch.no_grad():
            y = net(text_idx)
            y = torch.argmax(F.softmax(y, dim=-1)).detach().numpy()
        
        category = str(y)
        return render_template('result.html', claim_1st = claim_1st, number = number, category=category)
    elif request.method == 'GET':
        return render_template('index.html')
        
if __name__ == '__main__':
    app.run()

