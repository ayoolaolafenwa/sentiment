import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten


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

model.summary()


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(train_x, train_y, validation_data = [test_x, test_y], epochs = 1, batch_size = 64)

accuracy = model.evaluate(test_x, test_y)

print(accuracy[1])

model.save("sent.h5")