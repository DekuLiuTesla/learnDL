import os
import time
import socket
import random
import requests
from bs4 import BeautifulSoup
from progressbar import *


def geturl(url):
    headers = {'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                             " ""Chrome/78.0.3904.70 Safari/537.36", }  # including infos about agent browser
    req = requests.get(url=url, headers=headers)
    req.encoding = "UTF-8"  # Encoding mode
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find("div", class_="pc_list")  # find url list for each chapter
    chapters = texts.find_all("a")  # label "a" records corresponding url for chapter in href propety
    websites = []
    # Extract url for each chapter
    for chapter in chapters:
        name = chapter.string
        href = chapter.get("href")
        url_chapter = href if url in href else url + href
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
            response.encoding = "UTF-8"  # deal with messy code
            soup = BeautifulSoup(response.text, 'lxml')
            texts = soup.find("div", kwargs)  # extract specific infos
            texts_list = texts.text.replace(u'\u3000', u''). \
                replace('\r', '').replace(" ", '').split("\xa0" * 4)
            with open(path, "w") as file:
                for line in texts_list:
                    file.write(line + "\n")
            response.close()
    except:
        time.sleep(5)
        get_txt(url, caption, headers, save_pth, **kwargs)


if __name__ == '__main__':
    # j = 0
    # renamed_dir = "./texts/DouLuoDaLu2_renamed/"
    save_dir = "./texts/DouLuoDaLu2/"
    target_url = 'https://www.xuanshu.com/book/23099/'  # target website
    targets = geturl(target_url)
    headers = {'user-agent': "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) "
                             "Gecko/20100101 Firefox/34.0"}  # including infos about agent browser
    pbar = ProgressBar().start()
    for i, tar in enumerate(targets):
        file_name = tar[1].replace(" ", '_') + ".txt"
        get_txt(tar[0], tar[1], headers, save_dir, id="content1")
        # if os.path.exists(save_dir + file_name):
        #     os.rename(save_dir + file_name, save_dir + renamed_dir + f'chapter_{j}' + ".txt")
        #     j += 1
        pbar.update(int(((i + 1) / len(targets)) * 100))
