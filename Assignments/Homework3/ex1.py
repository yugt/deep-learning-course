import nltk
# nltk.download('punkt')

with open('data_HW3_Plato_Republic.txt') as myfile:
    myText = myfile.read().replace('\n', ' ')
    myText = myText.encode('ascii', 'ignore')
    myText = myText.decode()

###### (a) ######
myText_tokenized = nltk.word_tokenize(myText.lower())
T = len(myText_tokenized)
print("-- length of texts (number of words T) : ", T)

###### (b) ######
myUnigram = nltk.ngrams(myText_tokenized, 1)
fdist_uni = nltk.FreqDist(myUnigram)
common = [word for word in fdist_uni.most_common() if len(word[0][0])>=8]
print(common[0:5])

###### (c) ######
myBigram = nltk.ngrams(myText_tokenized, 2)
fdist_bi = nltk.FreqDist(myBigram)
def fcn(w1, w2):
    return fdist_bi[(w1, w2)]/fdist_uni[(w1,)]

###### (d) ######
perplexity = 1.0
for k in range(T-1):
    perplexity *= fcn(myText_tokenized[k], myText_tokenized[k+1])**(-1/(T-1))
print(perplexity)