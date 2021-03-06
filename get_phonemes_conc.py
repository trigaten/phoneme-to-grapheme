import os
import re
import numpy as np
import bs4
import codecs
import requests
import time
import concurrent.futures  # This import is important for concurrency.

start_time = time.time()
codecs.register_error("strict", codecs.ignore_errors)

# Basic Background Info
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
merriam = "https://www.merriam-webster.com/dictionary/"
dictionary_web = "https://www.dictionary.com/browse/"
base_url = dictionary_web  # URL to get words and phonetics from


dict_filename = "words_beta.txt"  # `Name of the file containing many words - error
err_filename = "404s.txt"  # List of all of the known error words

# Get current location based on operating system
fileDir = os.path.dirname(os.path.realpath('words_beta.txt'))

original_size = 0  # Change this value to change the dataset size
new_size = original_size  # Used to subtract the size of any existing sets from the amount needed.

# Do the initial parsing of the dictionary filev
dict_file = open(dict_filename).read()
dict_list = dict_file.split("\n")
curr_words = []
dict_len = len(dict_list)

regex = r"( |\n([a-z][A-Z])*\n|\'|\[|\]|ˈ|\+|\"|\(" \
        r"|\)|ˌ||-|͟|¦|\||‧|͟|&|1|2|–|—|͟|‧|;|pronunciationat|\r|\\|\/|for\d*|\d)*"
to_replace = r"^noun |^pronoun |^verb |^adjective |^adverb |^preposition |^conjunction |^interjection "

write_file = "phonemes-words.csv"  # The file to write phoneme-words to
append_or_write = "a"  # w to erase write_file and rewrite, a to append to current csv

words_added = 0  # Used to keep track of the total number of words added
fixed_set = []  # Set of words added to the phonetics

# Separate the file by lines to retrieve the length
if append_or_write == "a":
    temp_read = codecs.open(write_file, "r", "utf-8-sig")
    new_size -= len(codecs.open(write_file, "r", "utf-8-sig").read().split("\n"))
    temp_read.close()


# This is the main function to get the urls of size "size"
def get_urls(size):
    global dict_list
    empty_set = set([None])  # Empty set used to remove empty sets from lists
    urls = [None] * size  # Indexing the urls using iterators is around 25% faster than appending

    # Keep adding numbers until the target size is reached
    print(len(dict_list))
    for i in range(size):
        new_num = np.random.randint(0, high=len(dict_list))  # Get random number
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
            regex1 = r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ||-|͟|¦|\||‧|͟|&|–|—|͟|‧|pronunciationat)*"
            web_result = curr_page.content  # Return the new word and page contents
            soup = bs4.BeautifulSoup(web_result, "html.parser")
            word_soup = soup.find_all('h1', {'class': 'hword'})
            phonetics = soup.find_all('span', {'class': 'pr'})

        else:
            regex1 = r"(ˈ| |/)"
            web_result = curr_page.content  # Return the new word and page contents
            soup = bs4.BeautifulSoup(web_result, "html.parser")
            word_soup = soup.find_all('h1', {'class': 'css-1sprl0b e1wg9v5m5'})
            phonetics = soup.find_all('span', {'class': 'pron-ipa-content css-7iphl0 evh0tcl1'})
            # print(str(word_soup) + "," + str(phonetics))

        # Check if both words are of valid lengths
        if len(phonetics) >= 1 and len(word_soup) >= 1:
            global new_size, total_failed, words_added  # Total number of words added

            actual_word = word_soup[0].text.lower()  # Make sure all text is lowercase

            if actual_word in curr_words:
                add_err()
                return
            words_added += 1
            phonetics = phonetics[0].text.lower()  # Retrieve phonetics

            # Some phonetics have multiple pronunciation variations, we only use one
            if "," or ";" in phonetics:
                phonetics = phonetics.split(",")[0].split(";")[0]
                # print("changed p: " + phonetics)

            # Remove all invalid characters

            phonetics = re.sub(to_replace, "", phonetics)
            fixed_phonetics = re.sub(regex1, "", phonetics)
            # Combine the words into one line for CSV preparation
            csv_formatted = fixed_phonetics + "," + actual_word

            # Write the line to the file.
            text_file = codecs.open(write_file, "a", "utf-8-sig")
            text_file.write(csv_formatted + "\n")
            text_file.close()

            print("Words Left:\t\t" + str(new_size - words_added) + "\t\t" + csv_formatted)

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
    read_file = codecs.open(write_file, "r", "utf-8-sig")
    fix_lines = read_file.read()
    read_file.close()

    init_length = len(fix_lines.split("\n"))

    global regex
    re2 = r"\n(noun|pronoun|verb|adjective|adverb|preposition|conjunction|interjection)"
    # print(re.findall(re2, fix_lines))
    fix_lines = re.sub(re2, "\n", fix_lines).replace("or,", ",")
    lines = re.sub(regex, "", fix_lines).replace(r"﻿", "").split("\n")

    new_lines = []
    drop_lines = []

    # Recreate the original object using csv
    for line in lines:
        split_lines = line.split(",")
        if len(split_lines) == 2:
            new_lines.append(split_lines)

        else:
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
    purged = init_length - len(final_set)
    # words_added -= purged

    print("Total_Purged:\t\t" + str(purged))

    end_block = codecs.open(write_file, "w", "utf-8-sig")
    end_block.write("phonemes,graphemes\n")

    for line in final_set:
        end_block.write(str(line).lower() + "\n")
    end_block.close()

    error_file = codecs.open(err_filename, "r", "utf-8-sig")
    err = set(error_file.read().replace("\r", "").split("\n"))
    error_file.close()

    global dict_list, curr_words
    curr_words = set([word.split(",")[1] for word in final_set])
    # print(curr_words)
    # updated dict fil
    new_len = set(dict_list) - curr_words
    print(str(len(dict_list)) + "," + str(len(new_len)) + "," + str(len(new_len - err)))
    dict_list = list((set(dict_list) - curr_words) - err)
    return final_set


times_phon_was_run = 0


# The main function used to request and process phonetics.
def phon(size):
    global new_size
    global times_phon_was_run
    global words_added, total_failed, dict_list, dict_file
    words_added = total_failed = 0
    group_size = 1000
    new_size = size

    times_phon_was_run += 1
    print("Again: " + str(times_phon_was_run))
    rounds = int(size/group_size)

    for i in range(rounds):
        print("Round: " + str(i))
        urls = get_urls(group_size)
        get_all(urls)
        remove_invalids()

    urls = get_urls(size - rounds * group_size)
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