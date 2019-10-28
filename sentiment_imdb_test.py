import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence


top_words = 5000
(train_x, train_y),(test_x, test_y) = imdb.load_data(num_words = top_words)

max_len = 500

#pad sequences to have sequences of the same length
train_x = sequence.pad_sequences(train_x, maxlen =  max_len)
test_x = sequence.pad_sequences(test_x, maxlen = max_len)

#dimension of embedding space
embed_dim = 32

model = Sequential()

model.add(Embedding (input_dim = top_words, output_dim = embed_dim, input_length = max_len))
model.add(LSTM(100))
model.add(Dense (1, activation = "sigmoid"))

#model.summary()

model.load_weights("sent.h5")
word_index = keras.datasets.imdb.get_word_index()

def index(word):
    if word in word_index:
        return word_index[word]
    else:
        return "0"

def sequences(words):
    words = text_to_word_sequence(words)
    seqs = [[index(word) for word in words if word != "0"]]
    return sequence.pad_sequences(seqs, maxlen=max_len)

bad_seq = sequences("I hate this moview")
good_seq = sequences("I love this movie")

print("bad movie: " + str(model.predict(bad_seq)))   
print("good movie: " + str(model.predict(good_seq)))
