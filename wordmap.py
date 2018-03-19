from hazm import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#Clean up first text

infile1 = "book1.txt"
outfile1 = "cleaned1.txt"

f=open("stop.txt")
delete_list=f.read().split("\n")
fin1 = open(infile1)
fout1 = open(outfile1, "w+")
for line in fin1:
    for word in delete_list:
        line = line.replace(word, "")
    #print(line)
    fout1.write(line)
#fin1.close()
#fout1.close()

#Clean up second text

infile2 = "book2.txt"
outfile2 = "cleaned2.txt"

f=open("stop.txt")
delete_list=f.read().split("\n")
fin2 = open(infile2)
fout2 = open(outfile2, "w+")
for line in fin2:
    for word in delete_list:
        line = line.replace(word, "")
    #print(line)
    fout2.write(line)
#fin2.close()
#fout2.close()

#Normalize first text

print('***************** NORMALIZATION *******************')
n = Normalizer()
for sample in outfile1:
    #print(sample)
    sample = n.normalize(sample)
    #print(sample)

print('************* SENTENCE TOKENIZATION ***************')
all_sentences = []
for sample in outfile1:
    sentences = sent_tokenize(sample)
    all_sentences.extend(sentences)
    #print(all_sentences)
print('**************** WORD TOKENIZATION *****************')
for sentence in all_sentences:
    sentence = word_tokenize(sentence)


#Normalize second text

print('***************** NORMALIZATION *******************')
n = Normalizer()
for sample in outfile2:
    #print(sample)
    sample = n.normalize(sample)
    #'print(sample)
    #print()


print('************* SENTENCE TOKENIZATION ***************')
all_sentences = []
for sample in outfile2:
    sentences = sent_tokenize(sample)
    all_sentences.extend(sentences)
    #print(all_sentences)
print('**************** WORD TOKENIZATION *****************')
for sentence in all_sentences:
    sentence = word_tokenize(sentence)

obj = TfidfVectorizer()

f1 = open("cleaned.txt")
f2 = open("cleaned2.txt")
doc1 = f1.read()
doc2 = f2.read()
corpus = [doc1, doc2]
X = obj.fit_transform(corpus)
print(X)

terms = obj.get_feature_names()


indices = np.argsort(obj.idf_)[::-1]
features = obj.get_feature_names()
top_n = 60
top_features = [features[i] for i in indices[:top_n]]
print(top_features)
