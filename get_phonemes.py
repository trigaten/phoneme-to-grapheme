import os
import requests as rq
import re
import numpy as np
import bs4
import codecs

#Basic Background Info
headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:42.0) Gecko/20100101 Firefox/42.0"}
url = "https://www.merriam-webster.com/dictionary/" #URL to get words and phonetics from
filename = "words_alpha.txt"
fileDir = os.path.dirname(os.path.realpath('words_alpha.txt'))

size_original = 1000 #Change this value to change the dataset size


dictionary_file = open("words_alpha.txt").read()
dictionary = dictionary_file.split("\n")
dict_len = len(dictionary)
all_phonetics_tuples = []

#This is the main function to get the html files of size "size"
def get_phonetics(size):

    rand_nums = [] #used to prevent the same word from being included in the set more than once
    while(len(rand_nums) < size):
        def get_dictionary_html():
            new_num = np.random.randint(0, high=dict_len)

            #Find a new number that is not already in the list of random numbers
            while new_num in rand_nums:
                print("Num changed")
                new_num = np.random.randint(0, high=dict_len)

            #Add the new, unique number to the rand_nums list
            word = dictionary[new_num]
            #changes the base url to include the new word to get that word from the website
            modified_url = url + word
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
                    phonetics = re.search(r"> *(\D)*<", match.group(0)).group(0)
                    phonetics = phonetics[1:len(phonetics) - 1]
                    phonetics = re.sub(r"( |\'|\[|\]|ˈ|\+|\"|\(|\)|ˌ|</span>|-)*", "", phonetics)

                    #Some words included a + in them, so we removed those.
                    all_phonetics_tuples.append([str(phonetics) + "," + str(word)])
                    rand_nums.append(new_num)
                    print("Added")

                except:
                    #Show that an error has occured
                    print("Something went wrong")
                return curr_page.content

        web_result = get_dictionary_html()

#Main function to get all of the phonetics
get_phonetics(size_original)

#Manually make CSV because none of the default libraries work for phonetics
csv = all_phonetics_tuples

#Export csv_str as a utf-8-sig formated file separated by ","
with codecs.open("phonemes-words.csv", "w", "utf-8-sig") as text_file:
    for line in csv:
        line = re.sub(r"(\[|\]|\"|\')*", "", str(line))
        text_file.write(line+"\n")

text_file.close()

