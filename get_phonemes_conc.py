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

# Basic Background Info
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
original_url = "https://www.merriam-webster.com/dictionary/"  # URL to get words and phonetics from
dictionary_name = "words_beta.txt"  # Location of the list of words used to retrieve phonetics
err_filename = "404s.txt"  # List of all of the known error words

# Get current location based on operating system
fileDir = os.path.dirname(os.path.realpath('words_beta.txt'))

# Change this value to change the dataset size
size_original = 30000

# Do the initial parsing of the dictionary file
dictionary_file = open(dictionary_name).read()
dictionary = dictionary_file.split("\n")
dict_len = len(dictionary)

# Where will the file be writing to?
write_file = "phonemes-words.csv"

# Change the "w" option to an "a" to append stringsd to the current file
# Change the "a" option to "w" to erase the file and write to a blank file.
append_or_write = "a"

words_added = 0  # Used to keep track of the total number of words added
fixed_set = []  # Set of words added to the phonetics


# This is the main function to get the html files of size "size"
def get_urls(size):
    empty_set = set([None])  # Empty set used to remove empty sets from lists
    rand_nums = [None] * size  # Used to prevent the same word from being included in the set more than once
    urls = [None] * size  # Indexing the urls using iterators is around 25% faster than appending

    # Keep adding numbers until the target size is reached
    for i in range(size - len(set(rand_nums) - empty_set)):
        new_num = np.random.randint(0, high=dict_len)  # Get random number
        rand_nums[i] = new_num  # Set random number using i as the index
        word = dictionary[new_num]  # Use random number to retrieve from dictionary

        # Changes the base url to include the new word to get that word from the website
        modified_url = original_url + word
        urls[i] = modified_url

    return_set = set(urls) - set([None])  # Remove empty set and duplicates from list of urls

    # Keep getting more urls until the size is reached.
    while len(return_set) < size:
        print("Return Set Length: " + len(return_set))
        return_set.update(get_urls(size - len(return_set)))

    return return_set


total_failed = 0  # Uses this variable to track number of items that fail.


# This method takes a html page and word as a parameter to parse and retrieve the word on the page and phonetics.
def get_word(curr_page, word):
    # A commonly used sequence of lines in this method to add the word to list of unavailable words
    def add_err():
        # Write the word to the not_found list
        not_found = codecs.open(err_filename, "a")
        not_found.write(word + "\n")
        not_found.close()

        global total_failed
        total_failed -= 1  # Decrement the total fails if the page is unavailable.
        print("Failed: " + word + " : " + str(total_failed) + "\n")

    if curr_page.status_code == 404:
        add_err()  #
        # print("Error 404")
        # If the page was not valid, try another number combination

    else:
        # Retrieve the word and phonetics from the page using bs4
        web_result = curr_page.content  # Return the new word and page contents
        soup = bs4.BeautifulSoup(web_result, "html.parser")
        word_soup = soup.find_all('h1', {'class': 'hword'})
        phonetics = soup.find_all('span', {'class': 'pr'})

        # Check if both words are of valid lengths
        if len(phonetics) >= 1 and len(word_soup) >= 1:
            global words_added  # Total number of words added
            words_added += 1
            print("Words Left : " + str(size_original - words_added))

            actual_word = word_soup[0].text.lower()  # Make sure all text is lowercase
            phonetics = phonetics[0].text  # Retrieve phonetics

            # Some phonetics have multiple pronunciation variations, we only use one
            if "," in phonetics:
                phonetics = phonetics.split(",")[0]

            # Remove all invalid characters
            fixed_phonetics = re.sub(r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧)*", "", phonetics)
            # Combine the words into one line for CSV preparation
            csv_formatted = fixed_phonetics + "," + actual_word

            # Write the line to the file.
            text_file = codecs.open(write_file, "a", "utf-8-sig")
            text_file.write(csv_formatted + "\n")
            text_file.close()

            return csv_formatted

        else:
            # If the adding part does succeed, then add word to error list
            add_err()


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


def remove_invalids():
    global total_failed
    new_fails = total_failed
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

    not_found = codecs.open(err_filename, "a")
    for remove in remove_lines:
        try:
            remove_set.add(str(remove[0]) + "," + str(remove[1]))
            not_found.write(str(remove[1]) + "\n")
            total_failed -= 1

        except:
            not_found.write(str(remove[0]) + "\n")
            total_failed -= 1

    not_found.close()
    print("Total_Purged: " + str(new_fails - total_failed))

    final_set = set(lines) - remove_set - set([""])

    end_block = codecs.open(write_file, "w", "utf-8-sig")
    for line in final_set:
        end_block.write(str(line) + "\n")
    end_block.close()

    return final_set


ran_i = 0


def phon(size):
    dictionary_file = open(dictionary_name).read()
    dictionary = dictionary_file.split("\n")
    dict_len = len(dictionary)

    global ran_i
    ran_i += 1
    print("Again: " + str(ran_i))
    urls = get_urls(size)
    get_all(urls)


if append_or_write == "a":
    size_original -= len(codecs.open(write_file, "r", "utf-8-sig").read().split("\n"))

remove_invalids()
phon(size_original + total_failed)
# Export csv_str as a utf-8-sig formated file separated by ","

text_file = codecs.open(write_file, append_or_write, "utf-8-sig")
not_found = codecs.open(err_filename, "a")
text_file.close()

fixed_set = remove_invalids()

# familypronunciation
# pronunciation

while len(fixed_set) < size_original:
    phon(size_original - len(fixed_set))
    fixed_set = remove_invalids()
    words_added = len(fixed_set)

# not_found.close()
total_time = time.time() - start_time
print(total_time)

time_file = codecs.open("conc_timing.txt", "a")
time_file.write(str(total_time) + "\n")
time_file.close()
not_found.close()

a_dict = set(codecs.open("words_alpha.txt", "r", "utf-8-sig").read().replace("\r", "").split("\n"))
err = set(codecs.open(err_filename, "r", "utf-8-sig").read().split("\n"))
b_write = codecs.open(dictionary_name, "w", "utf-8-sig")

new_dict_set = a_dict - err

for word in new_dict_set:
    b_write.write(word + "\n")

b_write.close()
