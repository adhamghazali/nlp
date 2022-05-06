
from process_data import process_tweet
import numpy as np
from bayes_utils import process_tweet,lookup

def train_naive_bayes(freqs, train_x, train_y):

    loglikelihood = {}
    logprior = 0


    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    N_pos = N_neg = 0
    V_pos = V_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            # print(freqs[pair])

            N_pos += freqs[pair]
            V_pos += 1
        else:

            N_neg += freqs[pair]
            V_neg += 1

    D = len(train_y)

    D_pos = np.count_nonzero(train_y == 1)  # (len(list(filter(lambda x: x > 0, train_y))))
    D_neg = np.count_nonzero(train_y == 0)
    logprior = np.log(D_pos ) - np.log(D_neg )

    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)

        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):

    word_l = process_tweet(tweet)

    p = 0

    p += logprior

    for word in word_l:

        if word in loglikelihood:
            p += loglikelihood[word]

    return p


def test_functionality():
    import process_data
    train_x,train_y,test_x,test_y,freqs=process_data.prepare_training_data()
    logprior, loglikelihood=train_naive_bayes(freqs,train_x,train_y)

    my_tweet = 'you are great, and fantastic but an idiot'
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print('output = ', p)
    if p < 0:
        print("Negative Sentiment")
    if p>0:
        print('Positive Sentiment')
    else:
        print("Neutral Sentiment")



if __name__ == "__main__":
    test_functionality()
