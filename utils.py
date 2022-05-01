#FE
import functools
import numpy as np

def constuct_vocab(list_of_sentences):
   vocab=[]
   used=set()
   for sentence in list_of_sentences:
      sentence=sentence.split(" ")
      vocab.extend(sentence)
   vocab=[x for x in vocab if x  not in used and (used.add(x) or True)]
   print(vocab)
   return vocab



def feature_extract(sentence,vocab):
   #sparse feature extractor
   sentence=sentence.split(" ")
   feature=[0]*(len(vocab)+1)
   for word in sentence:
      index = vocab.index(word)
      feature[index]=1
   return feature


def frequency_features(tweets,labels,vocab):
   #sentence=sentence.split(" ")
   labels=np.array(labels)
   pos=np.where(labels == 1)
   neg=np.where(labels ==0)
   positive_list=[tweets[i] for i in pos[0]]
   positive_list=[x.split(" ") for x in positive_list]

   negative_list=[tweets[i] for i in neg[0]]
   negative_list = [x.split(" ") for x in negative_list]
   pos_freq=[0]*(len(vocab)+1)
   neg_freq=[0]*(len(vocab)+1)
   count=0
   for word in vocab:
      print(word)
      print(positive_list)
      if word in positive_list:
         print("positive condition")
         pos_freq[count]+=1
      if word in negative_list:
         neg_freq[count]+=1

   return pos_freq,neg_freq


def test_functionlity():
   tweets=["I love this company","this is a bad movie","do you always run that slow","I am the sun boy" ]
   labels=[1,0,0,1]

   vocab=constuct_vocab(tweets)
   feature=feature_extract(tweets[0],vocab)

   pos_freq, neg_freq=frequency_features(tweets,labels,vocab)
   #print("Feature", feature)
   print(pos_freq,neg_freq)





if __name__ == "__main__":
    test_functionlity()