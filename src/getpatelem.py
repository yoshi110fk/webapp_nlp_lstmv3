from bs4 import BeautifulSoup as bs4
import requests

def getclaim1(pn):
    # Google Patentsのベースアドレス
    url = 'https://patents.google.com/patent/'

    res = requests.get(url + pn)
    if res.status_code != requests.codes.OK:
        pn = pn + '1'
        res = requests.get(url + pn)   
    soup = bs4(res.text, "html.parser")
    
    # type0の場合
    if len(soup.select('#CLM-00001')) != 0:
        claim_1st = soup.select('#CLM-00001')[0].get_text().strip()
        claim_1st = claim_1st.replace('\n', ' ')
        claim_1st = claim_1st.replace('1. ', '')

    # type1の場合
    elif soup.find(num="1") != None:
        claim_1st = soup.find(num="1").get_text().strip()
        claim_1st = claim_1st.replace('\n', ' ')
        claim_1st = claim_1st.replace('1. ', '')
   
    # type0, type1以外で抽出できない場合
    else:
        claim_1st = "not extracted"
    return claim_1st