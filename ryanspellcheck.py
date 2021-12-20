import random
import codecs
from spellchecker import SpellChecker

dict_filename = "words_beta.txt"  # `Name of the file containing many words - error
err_filename = "404s.txt"  # List of all of the known error words

dict_file = open(dict_filename).read()
dict_list = dict_file.split("\n")

a_dict = set(codecs.open("words_alpha.txt", "r", "utf-8-sig").read().replace("\r", "").split("\n"))
# b_write = codecs.open(dict_filename, "w", "utf-8-sig")
err = set(codecs.open(err_filename, "r", "utf-8-sig").read().replace("\r", "").split("\n"))

new_dict_set = list(a_dict - err)  # Subtract the set of errors from the beta dictionary


spellcheck = SpellChecker().correction
rand_words = ["test", "tes", "roar", "ror", "hower", "ower", "hour", "gold", "gowld"]
correct_list = ["test", "test", "roar", "roar", "hour", "hour", "hour", "gold", "gold"]
#pass this a list of word string outputs to spellcheck and correct all of them

def spellcheck_all(str_list):
    new_list = [None] * len(str_list)
    str_list = list(str_list)

    for i in range(len(str_list)):
        new_list[i] = spellcheck_1(str_list[i])

    print("original list: " + str(new_list))
    return new_list

def spellcheck_1(str):
    spellchecked = str
    if not str in a_dict:
        spellchecked = (spellcheck("".join(str)))
        print("new_word: " + spellchecked)
    return spellchecked