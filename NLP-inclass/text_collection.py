import requests
from bs4 import BeautifulSoup

headers = {'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                         " ""Chrome/78.0.3904.70 Safari/537.36", }  # including infos about agent browser
url = 'http://www.quanben.io/n/douluodalu/1.html'  # target website

if __name__ == '__main__':
    response = requests.request("GET", url, headers=headers)
    response.encoding = response.apparent_encoding  # deal with messy code
    soup = BeautifulSoup(response.text, 'lxml')
    texts = soup.find("div", id="content")  # extract specific infos
    texts_list = texts.text.split("\xa0" * 4)
    print(texts_list)
