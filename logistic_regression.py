
from process_data import process_tweet
import numpy as np

def sigmoid(z):

    return  1.0 / (1 + np.exp(-1*np.array(z)))

def extract_features(tweet, freqs, process_tweet=process_tweet):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    # bias term
    x[0, 0] = 1

    for word in word_l:
        if (word, 1.0) in freqs:
            x[0, 1] += freqs[(word, 1.0)]
        if (word, 0) in freqs:
            x[0, 2] += freqs[(word, 0.0)]

    assert (x.shape == (1, 3))
    return x


def gradientDescent(x, y, theta, alpha, num_iters):
    m,_=x.shape


    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        yt = np.transpose(y)
        J1 = np.dot(yt, np.log(h))
        J2 = np.dot((1 - yt), np.log(1 - h))
        J = (-1.0 / m) * (J1 + J2)
        theta = theta - ((alpha / m) * np.dot(np.transpose(x), h - y))

    J = float(J)
    return J, theta

def train_model(train_x,train_y,freqs):
    # collect the features 'x' and stack them into a matrix 'X'
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        X[i, :] = extract_features(train_x[i], freqs)

    # training labels corresponding to X
    Y = train_y

    # Apply gradient descent
    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
    return J,theta


def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return y_pred

def test_functionality():
    import process_data
    train_x,train_y,test_x,test_y,freqs=process_data.prepare_training_data()
    J,theta=train_model(train_x,train_y,freqs)





if __name__ == "__main__":
    test_functionality()
