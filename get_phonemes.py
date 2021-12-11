import os
import requests as rq
import re
import numpy as np
import bs4
import codecs
import time

start_time = time.time()

# Basic Background Info
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
url = "https://www.merriam-webster.com/dictionary/"  # URL to get words and phonetics from
dict_filename = "words_beta.txt"
fileDir = os.path.dirname(os.path.realpath('words_alpha.txt'))
phon_file_name = "old-phonemes.csv"

size_original = 100  # Change this value to change the dataset size

dictionary_file = open(dict_filename).read()
dictionary = dictionary_file.split("\n")
dict_len = len(dictionary)
all_phonetics_tuples = []

# This is the main function to get the html files of size "size"
text_file = codecs.open(phon_file_name, "w", "utf-8-sig")
not_found = codecs.open("404s.txt", "a", "utf-8-sig")


def get_phonetics(size):
    text_file = codecs.open(phon_file_name, "a", "utf-8-sig")
    rand_nums = []  # used to prevent the same word from being included in the set more than once

    while (len(rand_nums) < size):
        new_num = np.random.randint(0, high=dict_len)
        # spans = []
        # Find a new number that is not already in the list of random numbers
        while new_num in rand_nums:
            print("Num changed")
            new_num = np.random.randint(0, high=dict_len)

        # Add the new, unique number to the rand_nums list
        word = dictionary[new_num]
        # changes the base url to include the new word to get that word from the website
        modified_url = url + word
        curr_page = rq.get(modified_url, headers=headers)

        # Check that the page is valid
        if curr_page.status_code == 404:
            print("Error 404")
            not_found.write(word + "\n")
            # If the page was not valid, try another number combination
            continue

        else:
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

                fixed_phonetics = re.sub(r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\|‧|͟|&|1|2|–|—|͟||\¦)*", "", phonetics)

                if len(str(fixed_phonetics)) >= 1:
                    return_csv = fixed_phonetics + "," + actual_word
                    # Export csv_str as a utf-8-sig formated file separated by ","
                    text_file.write(return_csv + "\n")
                    all_phonetics_tuples.append(return_csv)
                    print(len(all_phonetics_tuples))
                    rand_nums.append(new_num)
    text_file.close()


# Main function to get all of the phonetics
get_phonetics(size_original)

# Manually make CSV because none of the default libraries work for phonetics
csv = all_phonetics_tuples

def remove_invalids():
    fix_lines = codecs.open(phon_file_name, "r", "utf-8-sig").read()
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

    end_block = codecs.open(phon_file_name, "w", "utf-8-sig")
    for line in final_set:
        end_block.write(str(line) + "\n")
    end_block.close()

    return final_set


fixed_set = remove_invalids()

#familypronunciation
#pronunciation

while len(fixed_set) < size_original:
    get_phonetics(size_original - len(fixed_set))
    fixed_set = remove_invalids()

total_time = time.time() - start_time
print(total_time)

time_file = codecs.open("timing.txt", "a")
time_file.write(str(total_time) + "\n")
time_file.close()

a_dict = set(codecs.open(dict_filename, "r", "utf-8-sig").read().split("\n"))
err = set(codecs.open("404s.txt", "r", "utf-8-sig").read().split("\n"))
b_write = codecs.open("words_beta.txt", "w", "utf-8-sig")

new_dict_set = a_dict - err
[b_write.write(word) for word in new_dict_set]
b_write.close()
