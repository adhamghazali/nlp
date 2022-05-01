#FE
import functools

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



def test_functionlity():
   tweets=["I love this company","this is a bad movie","do you always run that slow","I am a sun" ]
   vocab=constuct_vocab(tweets)
   feature=feature_extract(tweets[0],vocab)
   print("Feature", feature)



if __name__ == "__main__":
    test_functionlity()