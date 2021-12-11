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
size_original = 30700

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

# Used to subtract the size of any existing sets from the amount we need to find.
new_size = size_original

# Separate the file by lines to retrieve the length
if append_or_write == "a":
    new_size -= len(codecs.open(write_file, "r", "utf-8-sig").read().split("\n"))

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
        print("Return Set Length: " + str(len(return_set)))
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

        global total_failed, words_added
        total_failed -= 1  # Decrement the total fails if the page is unavailable.
        print("Failed:\t\t\t" + str("\t") + ":\t\t" + word)

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
            global words_added
            global new_size
            global total_failed  # Total number of words added
            words_added += 1

            actual_word = word_soup[0].text.lower()  # Make sure all text is lowercase
            phonetics = phonetics[0].text  # Retrieve phonetics

            # Some phonetics have multiple pronunciation variations, we only use one
            if "," in phonetics:
                phonetics = phonetics.split(",")[0]

            # Remove all invalid characters
            fixed_phonetics = re.sub(r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧|pronunciationat)*", "",
                                     phonetics)
            # Combine the words into one line for CSV preparation
            csv_formatted = fixed_phonetics + "," + actual_word

            # Write the line to the file.
            text_file = codecs.open(write_file, "a", "utf-8-sig")
            text_file.write(csv_formatted + "\n")
            text_file.close()

            print("Words Left:\t\t" + str(new_size - words_added - total_failed))
            return csv_formatted  # return the formatted string line

        else:
            # If the adding part does succeed, then add word to error list
            add_err()


import concurrent.futures  # This import is important for concurrency.


# This method is used to equest and process one url at a time
def get_one(url):
    curr_page = requests.get(url)
    get_word(curr_page, url.replace(original_url, ""))
    return curr_page.raise_for_status()


#   The concurrent method used to retrieve all urls at a maximum rate of 20 requests
def get_all(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_one, url) for url in urls]

    # Log the results and exceptions after the process is completed
    for fut in futures:
        if fut.exception() is not None:
            print('{}: {}'.format(fut.exception(), 'ERR'))
        else:
            print('{}: {}'.format(fut.result(), 'OK'))

#   Will write the notes for this part and below later
def remove_invalids():
    # global total_failed, words_added, new_size
    new_fails = total_failed
    fix_lines = codecs.open(write_file, "r", "utf-8-sig").read()
    lines = re.sub(r"(\n([a-z][A-Z])*\n)", "\n", fix_lines).replace(r"﻿", "").split("\n")
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

        except:
            try:
                not_found.write(str(remove[0]) + "\n")
            except:
                print("This did not work")

    not_found.close()

    final_set = set(lines) - remove_set - set([""])

    purged = new_fails - total_failed
    # words_added -= purged

    print("Total_Purged:\t\t" + str(purged))

    end_block = codecs.open(write_file, "w", "utf-8-sig")
    for line in final_set:
        end_block.write(str(line) + "\n")
    end_block.close()

    return final_set


fixed_set = remove_invalids()  # Removes around 99.9% of invalid lines
times_phon_was_run = 0


# The main function used to request and process phonetics.
def phon(size):
    global new_size
    global times_phon_was_run
    global words_added, total_failed
    words_added = total_failed = 0

    new_size = size
    dictionary_file = open(dictionary_name).read()
    dictionary = dictionary_file.split("\n")

    times_phon_was_run += 1
    print("Again: " + str(times_phon_was_run))
    urls = get_urls(size)
    get_all(urls)


phon(new_size + total_failed)  # Run the main function
fixed_set = remove_invalids()  # Remove invalids returns all items currently in the phonetics list

# Keep repeating the process of getting phoneme_words until all phonetics are reached.
while len(fixed_set) < size_original:
    phon(size_original - len(fixed_set))
    fixed_set = remove_invalids()

# Log the total time the function took to run.
total_time = time.time() - start_time
print(total_time)

# Store the time in the timing document
time_file = codecs.open("conc_timing.txt", "a")
time_file.write(str(total_time) + "\n")
time_file.close()

# Update the dictionary by subtracting all words that do not work.
a_dict = set(codecs.open("words_alpha.txt", "r", "utf-8-sig").read().replace("\r", "").split("\n"))
b_write = codecs.open(dictionary_name, "w", "utf-8-sig")
err = set(codecs.open(err_filename, "r", "utf-8-sig").read().replace("\r", "").split("\n"))
err_write = codecs.open(err_filename, "w", "utf-8-sig")

new_dict_set = a_dict - err  # Subtract the set of errors from the beta dictionary

# Write each line back to the file
for word in new_dict_set:
    b_write.write(word + "\n")

for errors in err:
    err_write.write(errors + "\n")

# Close out the file
b_write.close()
err_write.close()
