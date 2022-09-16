import os
import time
import socket
import random
import requests
from bs4 import BeautifulSoup
from progressbar import *


def get_url(url):
    headers = {'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                             " ""Chrome/78.0.3904.70 Safari/537.36", }  # including infos about agent browser
    req = requests.get(url=url, headers=headers)
    req.encoding = "UTF-8"  # Encoding mode
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find("div", class_="chapter_list")  # find url list for each chapter
    chapters = texts.find_all("a")  # label "a" records corresponding url for chapter in href propety
    websites = []
    # Extract url for each chapter
    for chapter in chapters:
        name = chapter.string
        href = chapter.get("href")
        url_chapter = href
        website = [url_chapter, name]  # stored in a list
        websites.append(website)
    return websites


def get_txt(url, caption, headers, save_pth, **kwargs):
    # Many retries to deal with ConnectionResetError: [WinError 10054]
    try:
        path = save_pth + caption.replace(" ", '_') + ".txt"
        if not os.path.exists(path):
            requests.adapters.DEFAULT_RETRIES = 5
            s = requests.session()
            s.keep_alive = False
            response = s.get(url, headers=headers)
            response.encoding = "gbk"  # deal with messy code.
            soup = BeautifulSoup(response.text, 'lxml')
            texts = soup.find("div", class_="chapter-lan-en toleft")  # extract specific infos
            texts_list = texts.text.split("\xa0" * 4)
            with open(path, "w") as file:
                for line in texts_list:
                    file.write(line.replace(u'\xa0', u'') + "\n")
            response.close()
    except:
        time.sleep(5)
        get_txt(url, caption, headers, save_pth, **kwargs)


if __name__ == '__main__':
    save_dir = "./texts/TheShawshankRedemption/"
    target_url = 'https://www.24en.com/novel/lizhi/the-shawshank-redemption.html'  # target website
    targets = get_url(target_url)
    headers = {'user-agent': "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) "
                             "Gecko/20100101 Firefox/34.0"}  # including infos about agent browser
    pbar = ProgressBar().start()
    for i, tar in enumerate(targets):
        file_name = f'chapter_{i}'
        get_txt(tar[0], file_name, headers, save_dir, class_="chapter-lan-en toleft")
        pbar.update(int(((i + 1) / len(targets)) * 100))
