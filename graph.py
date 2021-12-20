import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("phonemes-words.csv")
#print(data["phonemes"])
dict = {}
for index, row in data.iterrows():
    g = row['graphemes']
    if len(g) in dict:
        dict[len(g)] += 1
    else:
        dict[len(g)] = 1
Y = []
for ele in dict.keys():
    Y.append(dict[ele])
X = dict.keys()
fig = plt.figure()
plt.bar(X, Y, 0.4, color="green")
plt.xlabel("* length")
plt.ylabel("numbers of * length")
plt.title("bar chart")
plt.savefig("word_length.jpg")

dict = {}
for index, row in data.iterrows():
            g = row['graphemes']
            lowString = g.lower()
            for char in lowString:
                if char in dict:
                    dict[char] += 1
                else:
                    dict[char] = 1
Y = []
for ele in dict.keys():
            Y.append(dict[ele]/len(data))
X = dict.keys()
fig = plt.figure()
plt.bar(X, Y, 0.4, color="green")
plt.xlabel("letter")
plt.ylabel("average letter in each word")
plt.title("bar chart")
plt.savefig("ave_letter.jpg")

def test_accuracy (real, model):
    accuracy = {}
    dict = {}
    for i in range(len(real)):
        if real[i] == model[i]:
           if len(real[i]) in accuracy:
               accuracy[len(real[i])] += 1
           else:
               accuracy[len(real[i])] = 1
        else:
            if len(real[i]) not in accuracy:
                accuracy[len(real[i])] = 0
        if len(real[i]) in dict:
            dict[len(real[i])] += 1
        else:
            dict[len(real[i])] = 1
    X = dict.keys()
    Y = []
    for i in X:
        Y.append(1.0*accuracy[i]/dict[i])
    fig = plt.figure()
    plt.bar(X, Y, 0.4, color="green")
    plt.xlabel("* length")
    plt.ylabel("accuracy of * length")
    plt.title("bar chart")
    plt.savefig("accuracy.jpg")


if __name__ == '__main__':
    real = ['apple', 'orange','banana','pineapple','peach','mango']
    model = ['app', 'orange','banana','watermelon','grape','lemon']
    test_accuracy(real, model)



