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

def spellcheck_all(str_list, correct_list):
    new_dict = [None] * len(str_list)
    total_equal = 0
    str_list = list(str_list)
    print(str_list)

    for i in range(len(str_list)):
        if not str_list[i] in a_dict:
            new_dict[i] = (spellcheck("".join(str_list[i])))
            print("new_word: " + new_dict[i])
        else:
            new_dict[i] = str_list[i]

        if new_dict[i] == str_list[i]:
            total_equal += 1

    print("original list: " + str(new_dict))
    difference = set(new_dict) - set(correct_list)
    return (difference, total_equal/len(str_list))

print(str(spellcheck_all(rand_words, correct_list)))