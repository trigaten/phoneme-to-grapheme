
# import pdfplumber
import os
import requests as rq
import re
import numpy as np
import bs4

headers = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"}

url = "https://www.merriam-webster.com/dictionary/" #URL to get words and phonetics from
filename = "words_alpha.txt"

fileDir = os.path.dirname(os.path.realpath('words_alpha.txt'))

size_original = 1000 #Change this value to change the dataset size

dictionary_file = open("words_alpha.txt").read()
dictionary = dictionary_file.split("\n")
dict_len = len(dictionary)
htmls = []
all_phonetics_tuples = [[]]

#This is the main function to get the html files of size "size"
def get_phonetics(size):

    rand_nums = np.random.randint(0, high=dict_len, size=size)

    #Get the numbers from the data
    for num in rand_nums:
        word = dictionary[num]
        modified_url = url + word
        curr_page = rq.get(modified_url, headers=headers)

        #Check that the page is valid before continuing
        if curr_page.status_code != 404:
            htmls.append(curr_page.content) #store HTML files just in case
            #
            soup = bs4.BeautifulSoup(curr_page.content, "html.parser")
            match = re.search("<span class=\"pr\"> *\S*<", str(soup))

            #Handle errors
            try:
                phonetics = re.search("> *(\D)+<", match.group(0)).group(0)
                phonetics = phonetics[1:len(phonetics) - 1].replace("-", "").replace("</span>", "")

                if not "+" in phonetics:
                    all_phonetics_tuples.append([phonetics, word])
                    print("added")

            except:
                print("Something is wrong here")

        else:
            #Returns the HTML
            def get_dictionary_html():
                new_num = np.random.randint(0, high=dict_len)

                #Find a number that is not in the set
                while new_num in rand_nums:
                    print("Num changed")
                    new_num = np.random.randint(0, high=dict_len)

                #changes the base url to include the new word to get that word from the website
                modified_url = url + dictionary[new_num]
                curr_page = rq.get(modified_url, headers=headers)

                #Check that the page is valid
                if curr_page.status_code == 404:
                    print("Error 404")
                    #If the page was not valid, try another number combination
                    return get_dictionary_html()

                else:
                    #Return the new word and page
                    web_result = curr_page.content
                    soup = bs4.BeautifulSoup(web_result, "html.parser")
                    match = re.search("<span class=\"pr\"> *\S+", str(soup))

                    #Not all of the pages were in the same format, so we remove potential errors
                    try:
                        #Use regex to retrieve the word from the html file
                        phonetics = re.search("> *(\D)*<", match.group(0)).group(0)
                        phonetics = phonetics[1:len(phonetics) - 1].replace("-", "").replace("</span>", "")

                        #Some words included a + in them, so we removed those.
                        if not "+" or "\"" in phonetics:
                            all_phonetics_tuples.append([phonetics, word])
                            print("Added def")

                    except:
                        #Show that an error has occured
                        print("something went wrong" )
                    return curr_page.content

            web_result = get_dictionary_html()

#Main sequence to get all of the phonetics
get_phonetics(size_original)

#Keep getting more phonetics until the size is reached
while len(all_phonetics_tuples) < size_original:
    size = size_original - len(all_phonetics_tuples)
    get_phonetics(size)