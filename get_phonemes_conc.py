import os
import re
import numpy as np
import bs4
import codecs
import asyncio
import concurrent.futures
import requests
import time
start_time = time.time()
#Basic Background InfoF
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
original_url = "https://www.merriam-webster.com/dictionary/" #URL to get words and phonetics from
dictionary_name = "words_beta.txt"
err_filename = "404s.txt"

fileDir = os.path.dirname(os.path.realpath('words_beta.txt'))

size_original = 100 #Change this value to change the dataset size


dictionary_file = open(dictionary_name).read()
dictionary = dictionary_file.split("\n")
dict_len = len(dictionary)
write_file = "phonemes-words.csv"

#change the "w" option to "a" to add more to the current file
#change  the "a" option to "w" to erase the file and start from scratch
text_file = codecs.open(write_file, "w", "utf-8-sig")
not_found = codecs.open(err_filename, "a", "utf-8-sig")

#Used to keep track of the total number of words added
int_total = []
fixed_set = []
#This is the main function to get the html files of size "size"
def get_urls(size):
    empty_set = set([None])
    rand_nums = [None] * size #used to prevent the same word from being included in the set more than once
    urls = [None] * size
    urls_len_multipier = 0

    for i in range(size - len(set(rand_nums) - empty_set)):
        new_num = np.random.randint(0, high=dict_len)
        rand_nums[i] = new_num
        word = dictionary[new_num]
        # changes the base url to include the new word to get that word from the website
        modified_url = original_url + word
        urls[i] = modified_url
        print(i)

    return_set = set(urls) - set([None])

    while len(return_set) < size:
        print(len(return_set))
        return_set.update(get_urls(size - len(return_set)))

    return return_set

def get_word(curr_page, word):

    if curr_page.status_code == 404:
        not_found = codecs.open(err_filename, "a", "utf-8-sig")
        not_found.write(word + "\n")
        not_found.close()
        print("Failed: " + word)
        return
        # print("Error 404")
        # If the page was not valid, try another number combination

    else:
        int_total.append("")
        print("Words Left : " + str(size_original - len(int_total)))
        # Return the new word and page
        web_result = curr_page.content
        soup = bs4.BeautifulSoup(web_result, "html.parser")
        word_soup = soup.find_all('h1', {'class': 'hword'})
        phonetics = soup.find_all('span', {'class': 'pr'})

        if len(phonetics) >= 1 and len(word_soup) >= 1:
            actual_word = word_soup[0].text.lower()
            phonetics = phonetics[0].text

            if "," in phonetics:
                phonetics = phonetics.split(",")[0]

            fixed_phonetics = re.sub(r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧)*", "", phonetics)
            return_word = fixed_phonetics + "," + actual_word
            text_file = codecs.open(write_file, "a", "utf-8-sig")
            text_file.write(return_word + "\n")
            text_file.close()

            return return_word

#Main function to get all of the phonetics

import concurrent.futures

def get_one(url):
    curr_page = requests.get(url)
    get_word(curr_page, url.replace(original_url, ""))
    return curr_page.raise_for_status()

def get_all(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_one, url) for url in urls]
    # the end of the "with" block will automatically wait
    # for all of the executor's tasks to complete

    for fut in futures:
        if fut.exception() is not None:
            print('{}: {}'.format(fut.exception(), 'ERR'))
        else:
            print('{}: {}'.format(fut.result(), 'OK'))

ran_i = []
def phon(size):
    ran_i.append("")
    print("Again: " + str(len(ran_i)))
    urls = get_urls(size)
    get_all(urls)

phon(size_original)
#Export csv_str as a utf-8-sig formated file separated by ","

text_file.close()

def remove_invalids():
    fix_lines = codecs.open(write_file, "r", "utf-8-sig").read()
    lines = fix_lines.replace("﻿", "").split("\n")
    new_lines = []

    for line in lines:
        new_lines.append(line.split(","))

    remove_lines = []
    for new_line in new_lines:
        if len(new_line) > 1 and len(new_line[0]) < (.5 * len(new_line[1])):
            remove_lines.append(new_line)

        elif len(new_line) <= 1:
            remove_lines.append(new_line)

    remove_set = set()
    remove_lines = remove_lines[:len(remove_lines) - 1]

    for remove in remove_lines:
        remove_set.add(str(remove[0]) + "," + str(remove[1]))

    final_set = set(lines) - remove_set - set([""])

    end_block = codecs.open(write_file, "w", "utf-8-sig")
    for line in final_set:
        end_block.write(str(line) + "\n")
    end_block.close()

    return final_set


fixed_set = remove_invalids()

#familypronunciation
#pronunciation

while len(fixed_set) < size_original:
    phon(size_original - len(fixed_set))
    fixed_set = remove_invalids()
    int_total = int_total[:len(fixed_set)]

# not_found.close()
total_time = time.time() - start_time
print(total_time)

time_file = codecs.open("conc_timing.txt", "a")
time_file.write(str(total_time) + "\n")
time_file.close()

a_dict = set(codecs.open("words_alpha.txt", "r", "utf-8-sig").read().replace("\r", "").split("\n"))
err = set(codecs.open(err_filename, "r", "utf-8-sig").read().split("\n"))
b_write = codecs.open(dictionary_name, "w", "utf-8-sig")

new_dict_set = a_dict - err

for word in new_dict_set:
    b_write.write(word + "\n")

b_write.close()
