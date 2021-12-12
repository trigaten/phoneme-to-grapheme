import os
import re
import numpy as np
import bs4
import codecs
import requests
import time
import concurrent.futures  # This import is important for concurrency.

start_time = time.time()

# Basic Background Info
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
merriam = "https://www.merriam-webster.com/dictionary/"
dictionary_web = "https://www.dictionary.com/browse/"
base_url = dictionary_web  # URL to get words and phonetics from

dict_filename = "words_beta.txt"  # `Name of the file containing many words - error
err_filename = "404s.txt"  # List of all of the known error words

# Get current location based on operating system
fileDir = os.path.dirname(os.path.realpath('words_beta.txt'))

original_size = 10000  # Change this value to change the dataset size
new_size = original_size  # Used to subtract the size of any existing sets from the amount needed.

# Do the initial parsing of the dictionary filev
dict_file = open(dict_filename).read()
dict_list = dict_file.split("\n")
dict_len = len(dict_list)

regex = r"( |\n([a-z][A-Z])*\n|\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧|;|pronunciationat|\r|\\|\/)*"

write_file = "phonemes-words.csv"  # The file to write phoneme-words to
append_or_write = "a"  # w to erase write_file and rewrite, a to append to current csv

words_added = 0  # Used to keep track of the total number of words added
fixed_set = []  # Set of words added to the phonetics

# Separate the file by lines to retrieve the length
if append_or_write == "a":
    new_size -= len(codecs.open(write_file, "r", "utf-8-sig").read().split("\n"))


# This is the main function to get the urls of size "size"
def get_urls(size):
    empty_set = set([None])  # Empty set used to remove empty sets from lists
    urls = [None] * size  # Indexing the urls using iterators is around 25% faster than appending

    # Keep adding numbers until the target size is reached
    for i in range(size):
        new_num = np.random.randint(0, high=dict_len)  # Get random number
        word = dict_list[new_num]  # Use random number to retrieve word from dictionary

        # Changes the base url to include the new word to get that word from the website
        modified_url = base_url + word
        urls[i] = modified_url

    return_set = set(urls) - empty_set  # Remove empty set and duplicates from list of urls

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
        if base_url is merriam:
            re1 = r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧|pronunciationat)*"
            web_result = curr_page.content  # Return the new word and page contents
            soup = bs4.BeautifulSoup(web_result, "html.parser")
            word_soup = soup.find_all('h1', {'class': 'hword'})
            phonetics = soup.find_all('span', {'class': 'pr'})

        else:
            re1 = r"(ˈ| |/)"
            web_result = curr_page.content  # Return the new word and page contents
            soup = bs4.BeautifulSoup(web_result, "html.parser")
            word_soup = soup.find_all('h1', {'class': 'css-1sprl0b e1wg9v5m5'})
            phonetics = soup.find_all('span', {'class': 'pron-ipa-content css-7iphl0 evh0tcl1'})
            # print(str(word_soup) + "," + str(phonetics))

        # Check if both words are of valid lengths
        if len(phonetics) >= 1 and len(word_soup) >= 1:
            global new_size, total_failed, words_added  # Total number of words added
            words_added += 1

            actual_word = word_soup[0].text.lower()  # Make sure all text is lowercase
            phonetics = phonetics[0].text  # Retrieve phonetics

            # Some phonetics have multiple pronunciation variations, we only use one
            if "," in phonetics:
                phonetics = phonetics.split(",")[0]

            # Remove all invalid characters

            fixed_phonetics = re.sub(re1, "",phonetics)
            # Combine the words into one line for CSV preparation
            csv_formatted = fixed_phonetics + "," + actual_word

            # Write the line to the file.
            text_file = codecs.open(write_file, "a", "utf-8-sig")
            text_file.write(csv_formatted + "\n")
            text_file.close()

            print("Words Left:\t\t" + str(new_size - words_added))

            # if words_added % 0 == 500:
            #     global dict_list, fixed_set
            #     dict_list = list(set(dict_list) - set(fixed_set))

            return csv_formatted  # return the formatted string line

        else:
            # If the adding part does succeed, then add word to error list
            add_err()


# This method is used to request and process one url at a time
def get_one(url):
    curr_page = requests.get(url)
    get_word(curr_page, url.replace(base_url, ""))
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
    new_fails = 0
    fix_lines = codecs.open(write_file, "r", "utf-8-sig").read()

    global regex
    lines = re.sub(regex, "", fix_lines).replace(r"﻿", "").split("\n")
    init_length = len(lines)
    new_lines = []
    drop_lines = []
    # Recreate the original object using csv
    for line in lines:
        split_lines = line.split(",")
        new_lines.append(split_lines)

        if len(split_lines) != 2:
            drop_lines.append(line)

    for line in drop_lines:
        lines.remove(line)
        print("removed: " + line)

    remove_lines = []

    #   Remove phonetics that looks suspicious by not being close to their word in length
    for new_line in new_lines:
        if len(new_line) == 2 and \
                (len(new_line[0]) < (.6 * len(new_line[1])) or len(new_line[0]) > (len(new_line[1]) + 3)):
            remove_lines.append(new_line)

        elif not len(new_line) == 2:
            remove_lines.append(new_line)

        else:
            non_abc = re.match(r"[a-zA-Z]*[^a-zA-Z\d\s:][a-zA-Z]*", new_line[1])
            if non_abc:
                print(non_abc.group(0))
                remove_lines.append(new_line)

    remove_set = set()
    remove_lines = remove_lines[:len(remove_lines) - 1]

    not_found = codecs.open(err_filename, "a")

    for remove in remove_lines:
        try:
            remove_set.add(str(remove[0]) + "," + str(remove[1]))
            not_found.write(str(remove[1]) + "\n")
            print("Removed: " + remove[1])
            new_fails += 1
        except:
            try:
                print(remove[0])
                remove_set.add(str(remove[0]))
                not_found.write(str(remove[0]) + "\n")
                new_fails += 1
            except:
                print("This did not work")
                error = codecs.open("weird_words.txt", "a", "utf-8-sig")
                error.write(str(remove))
                error.close()

    not_found.close()

    final_set = set(lines) - remove_set - set([""]) - set(["phonemes,graphemes\n"])

    purged = new_fails
    # words_added -= purged

    print("Total_Purged:\t\t" + str(purged))

    end_block = codecs.open(write_file, "w", "utf-8-sig")
    end_block.write("phonemes,graphemes\n")
    for line in final_set:
        end_block.write(str(line) + "\n")
    end_block.close()

    global dict_list
    dict_list = list(set(dict_list) - final_set)
    return final_set


times_phon_was_run = 0


# The main function used to request and process phonetics.
def phon(size):
    global new_size
    global times_phon_was_run
    global words_added, total_failed
    words_added = total_failed = 0

    new_size = size
    dictionary_file = open(dict_filename).read()
    dictionary = dictionary_file.split("\n")

    times_phon_was_run += 1
    print("Again: " + str(times_phon_was_run))
    urls = get_urls(size)
    get_all(urls)


fixed_set = remove_invalids()  # Removes around 99.9% of invalid lines
phon(new_size + total_failed)  # Run the main function
fixed_set = remove_invalids()  # Remove invalids returns all items currently in the phonetics list

# Keep repeating the process of getting phoneme_words until all phonetics are reached.
while len(fixed_set) < original_size:
    phon(original_size - len(fixed_set))
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
b_write = codecs.open(dict_filename, "w", "utf-8-sig")
codecs.register_error("strict", codecs.ignore_errors)
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
